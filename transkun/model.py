import math
import numpy as np
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from torch_log_wmse import LogWMSE


# ============================
#   Config（必要項目のみ）
# ============================
class ModelConfig:
    def __init__(self):
        # セグメント設定（学習・推論で参照）
        self.segmentHopSizeInSecond = 8
        self.segmentSizeInSecond = 16

        # STFT / オーディオ設定
        self.hop_length = 1024
        self.window_size = 2048
        self.fs = 44100
        self.num_channels = 2

        # 特徴量/モデル設定
        self.num_bands = 60
        self.hidden_size = 64
        self.num_heads = 8
        self.num_layers = 6
        self.hidden_factor = 4
        self.drop_prob = 0.1
        self.band_split_type = "bs"
        self.use_mixture_consistency = False

        # ロス重み
        self.loss_spec_weight = 1.0  # （トレーナ側で使用）
        self.loss_wmse_weight = 0.1

    def __repr__(self):
        return repr(self.__dict__)


Config = ModelConfig


# ============================
#   Multi-Resolution STFT Loss
# ============================
def _stft(audio: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    return torch.stft(audio, n_fft=n_fft, hop_length=hop, win_length=win.numel(), window=win, return_complex=True)


def _mrstft_sc_logmag(
    recon: torch.Tensor, target: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    1解像度ぶんの MR-STFT: Spectral Convergence + Log-Magnitude L1
    recon/target: [B, T]
    """
    spec_hat = _stft(recon, n_fft, hop, win)
    with torch.no_grad():
        spec_ref = _stft(target, n_fft, hop, win)

    mag_hat = torch.abs(spec_hat)
    mag_ref = torch.abs(spec_ref)

    # Spectral Convergence（振幅正規化）
    num = torch.linalg.vector_norm(mag_hat - mag_ref)
    den = torch.linalg.vector_norm(mag_ref) + eps
    loss_sc = num / den

    # Log-magnitude L1（広いダイナミクスで安定）
    loss_log = F.l1_loss(torch.log(mag_hat + eps), torch.log(mag_ref + eps))

    return loss_sc + loss_log


def multi_resolution_stft_loss_amplitude_invariant(
    recon_audio: torch.Tensor,  # [B, C, N, T]
    target_audio: torch.Tensor,  # [B, C, N, T]
    resolutions: List[int] = [4096, 2048, 1024, 512, 256],
    hop_length: int = 147,
    window_fn=torch.hann_window,
    stem_weights: Optional[List[float]] = None,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    b, c, n, t = recon_audio.shape
    device, dtype = recon_audio.device, recon_audio.dtype

    if stem_weights is None:
        stem_weights = [1.0] * n
    stem_w = torch.tensor(stem_weights, device=device, dtype=dtype)
    w_sum = stem_w.sum().clamp_min(1e-8)

    # 各winをキャッシュ（再生成を避ける）
    win_cache = {wl: window_fn(wl, device=device, dtype=dtype) for wl in resolutions}

    total = recon_audio.new_tensor(0.0)
    for stem_idx in range(n):
        w_stem = stem_w[stem_idx]
        for ch in range(c):
            recon_bt = recon_audio[:, ch, stem_idx].reshape(b, t)
            target_bt = target_audio[:, ch, stem_idx].reshape(b, t)
            for wl in resolutions:
                win = win_cache[wl]
                # checkpoint対応（メモリ節約）
                loss_i = torch.utils.checkpoint.checkpoint(
                    _mrstft_sc_logmag, recon_bt, target_bt, wl, hop_length, win, use_reentrant=False
                )
                total = total + w_stem * loss_i

    return (total / (c * w_sum)) * loss_weight


# ============================
#   Model (MSS専用)
# ============================
class TransKun(nn.Module):
    Config = ModelConfig

    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.conf = conf

        # 基本設定
        self.fs = conf.fs
        self.hop_size = conf.hop_length
        self.window_size = conf.window_size
        self.num_channels = conf.num_channels
        self.n_fft = self.window_size

        # ステム設定（2stem: [target, other]）
        self.num_stems = 2
        self.loss_wmse_weight = conf.loss_wmse_weight
        self.stem_weights = getattr(conf, "stem_weights", [1.0, 1.0])

        print("use_mixture_consistency:", conf.use_mixture_consistency)

        # バックボーン
        self.backbone = Backbone(
            sampling_rate=conf.fs,
            num_channels=conf.num_channels,
            n_fft=self.n_fft,
            hop_size=conf.hop_length,
            n_bands=conf.num_bands,
            hidden_size=conf.hidden_size,
            num_heads=conf.num_heads,
            ffn_hidden_size_factor=conf.hidden_factor,
            num_layers=conf.num_layers,
            dropout=conf.drop_prob,
            band_split_type=conf.band_split_type,
            use_mixture_consistency=conf.use_mixture_consistency,
        )

    # ---- ユーティリティ ----
    def getDevice(self):  # 後方互換
        return next(self.parameters()).device

    def _process_frames_batch(self, inputs_bcT: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_bcT: [B, C, T]
        Returns:
            recon_audio: [B, C, N, T]
        """
        recon_audio = self.backbone(inputs_bcT)
        return recon_audio

    # ---- 推論API ----
    @torch.no_grad()
    def separate(self, audio_slices: torch.Tensor) -> torch.Tensor:
        """ステム推定を返す簡易推論API。
        Args:
            audio_slices: [B, T, C]
        Returns:
            estimates: [B, T, 2, C]  (順序: 0=target, 1=other)
        """
        if audio_slices.ndim != 3:
            raise ValueError("audio_slices must be [B, T, C]")
        x_bcT = audio_slices.transpose(-1, -2)  # [B, C, T]
        recon_bcnT = self._process_frames_batch(x_bcT)  # [B, C, N, T]
        if recon_bcnT.shape[2] != self.num_stems:
            raise RuntimeError(f"expected {self.num_stems} stems, got {recon_bcnT.shape[2]}")
        # [B, C, N, T] -> [B, T, N, C]
        estimates = recon_bcnT.permute(0, 3, 2, 1).contiguous()
        return estimates

    def forward(self, audio_slices: torch.Tensor) -> torch.Tensor:
        """forwardはseparateと同じ挙動に統一（[B,T,2,C]）。"""
        return self.separate(audio_slices)

    @torch.no_grad()
    def separate_overlap(
        self,
        audio: torch.Tensor | np.ndarray,
        step_in_second: float | None = None,
        segment_size_in_second: float | None = None,
        use_hann: bool = True,
        return_numpy: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """
        長尺オーディオを Overlap-Add で分離するユーティリティ（固定仕様版）。
        - 推論前処理: 入力ミックスを LUFS -14 に「1回だけ」正規化（pyloudnorm/BS.1770）
        - 推論後処理: 出力ステムすべてに逆ゲインを掛け、元の音量へ復元
        - 3者（mixture/target/other）で共通係数のため SNR は不変

        Args:
            audio: [T, C] / [C, T] / [T]（float推奨）
            step_in_second: セグメントのホップ長（秒）。未指定時は conf.segmentHopSizeInSecond
            segment_size_in_second: セグメント長（秒）。未指定時は conf.segmentSizeInSecond
            use_hann: True なら OLA の重み付けに Hann 窓
            return_numpy: True のとき numpy で返す
        Returns:
            estimates: [T, N(=2), C]
        """

        # -------- 入力を numpy [T, C] にそろえる（まだGPUに載せない）--------
        def to_numpy_time_channel(x: torch.Tensor | np.ndarray) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
            else:
                if not np.issubdtype(x.dtype, np.floating):
                    x = x.astype(np.float32)
            if x.ndim == 1:
                x = x[:, None]  # [T, 1]
            # [C, T] 形式の可能性に配慮（通常は T のほうが長い）
            if x.shape[0] in (1, 2) and x.shape[1] > x.shape[0]:
                x = x.T
            return x.astype(np.float32, copy=False)

        audio_tc_np = to_numpy_time_channel(audio)
        sample_rate = self.fs

        # -------- 前処理: LUFS -14 へ 1回だけ正規化（係数を保持）--------
        try:
            import pyloudnorm as pyln
        except Exception as exc:
            raise RuntimeError("推論時の LUFS 正規化には pyloudnorm が必要です（pip install pyloudnorm）。") from exc

        loudness_meter = pyln.Meter(sample_rate)
        mono_for_meter = audio_tc_np.mean(axis=1)
        measured_lufs = float(loudness_meter.integrated_loudness(mono_for_meter))

        if np.isfinite(measured_lufs):
            gain_db = -14.0 - measured_lufs
            gain_linear = float(10.0 ** (gain_db / 20.0))
        else:
            # 無音などで LUFS が計算不能な場合はゲイン=1.0
            gain_linear = 1.0

        audio_tc_np = (audio_tc_np * gain_linear).astype(np.float32, copy=False)

        device = self.getDevice()
        audio_tc = torch.from_numpy(audio_tc_np).to(device)  # [T, C]
        audio_ct = audio_tc.transpose(0, 1)  # [C, T]

        if step_in_second is None:
            step_in_second = self.conf.segmentHopSizeInSecond
        if segment_size_in_second is None:
            segment_size_in_second = self.conf.segmentSizeInSecond

        hop_size = self.hop_size
        segment_size = int(math.ceil(segment_size_in_second * sample_rate))
        step_size = int(math.ceil(step_in_second * sample_rate / hop_size) * hop_size)
        pad_samples = max(segment_size - step_size, 0)

        if pad_samples > 0:
            audio_ct = F.pad(audio_ct, (pad_samples, pad_samples))
        total_length = audio_ct.shape[-1]

        estimates_nct = torch.zeros(
            self.num_stems, self.num_channels, total_length, device=device, dtype=audio_ct.dtype
        )
        window_sum_nct = torch.zeros_like(estimates_nct)

        use_rect_window = step_size >= segment_size
        if use_rect_window or not use_hann:
            window_1d = torch.ones(segment_size, device=device, dtype=audio_ct.dtype)
        else:
            window_1d = torch.hann_window(segment_size, device=device, dtype=audio_ct.dtype)
        window_nct = window_1d[None, None, :]  # [1, 1, L]

        for start_idx in range(0, total_length, step_size):
            end_idx = min(start_idx + segment_size, total_length)
            segment_ct = audio_ct[:, start_idx:end_idx]  # [C, L]
            current_len = segment_ct.shape[-1]
            if current_len < segment_size:
                segment_ct = F.pad(segment_ct, (0, segment_size - current_len))

            segment_tc = segment_ct.transpose(0, 1).unsqueeze(0)  # [1, T, C]
            estimates_btnc = self.separate(segment_tc)  # [1, T, N, C]
            estimates_seg_nct = estimates_btnc[0].permute(1, 2, 0)  # [N, C, T]

            valid_len = end_idx - start_idx
            win = window_nct[..., : estimates_seg_nct.shape[-1]]
            estimates_nct[..., start_idx:end_idx] += estimates_seg_nct[..., :valid_len] * win[..., :valid_len]
            window_sum_nct[..., start_idx:end_idx] += win[..., :valid_len]

        valid_mask = window_sum_nct > 0
        estimates_nct[valid_mask] = estimates_nct[valid_mask] / window_sum_nct[valid_mask]

        if pad_samples > 0:
            estimates_nct = estimates_nct[..., pad_samples:-pad_samples]

        # -------- 後処理: 逆ゲインで元音量に復元（全ステム同一係数）--------
        if gain_linear != 1.0:
            inverse_gain = 1.0 / gain_linear
            estimates_nct = estimates_nct * inverse_gain

        # [N, C, T] -> [T, N, C]
        outputs_tnc = estimates_nct.permute(2, 0, 1).contiguous()

        if return_numpy:
            return outputs_tnc.detach().cpu().numpy()
        return outputs_tnc

    # ---- 学習用ロス計算 ----
    def calc_loss(self, audio_slices: torch.Tensor, target_audio: torch.Tensor | None = None):
        """
        MSS用のロスを返す。
        Args:
            audio_slices: [B, T, C]
            target_audio: [B, T, N(=2), C] or None
        Returns:
            (loss_spec, loss_wmse)
        """
        x_bcT = audio_slices.transpose(-1, -2)  # [B, C, T]
        recon_bcnT = self._process_frames_batch(x_bcT)  # [B, C, N, T]

        loss_spec = torch.tensor(0.0, device=x_bcT.device)
        loss_wmse = torch.tensor(0.0, device=x_bcT.device)

        if target_audio is not None:
            # [B, T, N, C] -> [B, C, N, T]
            target_bcnT = target_audio.permute(0, 3, 2, 1)
            # 長さ合わせ（安全）
            t_len = min(target_bcnT.shape[-1], recon_bcnT.shape[-1])
            target_bcnT = target_bcnT[..., :t_len]
            recon_bcnT = recon_bcnT[..., :t_len]

            # Spec L1
            loss_spec = multi_resolution_stft_loss_amplitude_invariant(
                recon_audio=recon_bcnT, target_audio=target_bcnT, stem_weights=self.stem_weights
            )

            # Log-WMSE（波形系）
            log_wmse = LogWMSE(
                audio_length=recon_bcnT.shape[-1] / self.fs,
                sample_rate=self.fs,
                return_as_loss=True,
            )
            # inputs: [B, C, T], recon/target: [B, C, N, T]
            loss_wmse = log_wmse(x_bcT[..., :t_len], recon_bcnT, target_bcnT)

        return loss_spec, loss_wmse
