import math
import numpy as np
from typing import List, Optional

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
        self.hopSize = 1024
        self.windowSize = 2048
        self.fs = 44100
        self.num_channels = 2

        # 特徴量/モデル設定
        self.num_bands = 60
        self.baseSize = 64
        self.nHead = 8
        self.nLayers = 6
        self.hiddenFactor = 4
        self.scoringExpansionFactor = 4
        self.contextDropoutProb = 0.1
        self.band_split_type = "bs"

        # ロス重み
        self.loss_spec_weight = 1.0  # （トレーナ側で使用）
        self.loss_wmse_weight = 0.1

    def __repr__(self):
        return repr(self.__dict__)


Config = ModelConfig


# ============================
#   Multi-Resolution STFT Loss
# ============================
@torch.no_grad()
def _stft_ref(target: torch.Tensor, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor):
    return torch.stft(target, n_fft, hop_length, win_length, window=window, return_complex=True)


def _stft_l1(
    recon: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """1解像度分のL1損失。checkpoint対応のため引数はTensor化。"""
    spec_hat = torch.stft(
        recon, int(n_fft.item()), int(hop_length.item()), int(win_length.item()), window=window, return_complex=True
    )
    with torch.no_grad():
        spec_ref = torch.stft(
            target,
            int(n_fft.item()),
            int(hop_length.item()),
            int(win_length.item()),
            window=window,
            return_complex=True,
        )
    return F.l1_loss(spec_hat, spec_ref)


def multi_resolution_stft_loss(
    recon_audio: torch.Tensor,  # [B, C, N, T]
    target_audio: torch.Tensor,  # [B, C, N, T]
    resolutions: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
    window_fn=torch.hann_window,
    loss_weight: float = 1.0,
    stem_weights: Optional[List[float]] = None,
    hop_divisor: int = 4,
) -> torch.Tensor:
    b, c, n, t = recon_audio.shape
    if stem_weights is None:
        stem_weights = [1.0] * n
    stem_weights_t = torch.as_tensor(stem_weights, device=recon_audio.device, dtype=recon_audio.dtype)
    weight_sum = stem_weights_t.sum()

    total_loss = recon_audio.new_tensor(0.0)
    hop_list = [r // hop_divisor for r in resolutions]

    for stem in range(n):
        w_stem = stem_weights_t[stem]
        for ch in range(c):
            recon = recon_audio[:, ch, stem].reshape(b, t)  # [B, T]
            target = target_audio[:, ch, stem].reshape(b, t)  # [B, T]
            for win_len, hop_length in zip(resolutions, hop_list):
                window = window_fn(win_len, device=recon.device, dtype=recon.dtype)
                loss_i = torch.utils.checkpoint.checkpoint(
                    _stft_l1,
                    recon,
                    target,
                    torch.tensor(win_len),
                    torch.tensor(hop_length),
                    torch.tensor(win_len),
                    window,
                    use_reentrant=False,
                )
                total_loss = total_loss + w_stem * loss_i

    return (total_loss / (c * weight_sum)) * loss_weight


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
        self.hop_size = conf.hopSize
        self.window_size = conf.windowSize
        self.num_channels = conf.num_channels
        self.n_fft = self.window_size

        # ステム設定（2stem: [target, other]）
        self.num_stems = 2
        self.loss_wmse_weight = conf.loss_wmse_weight
        self.stem_weights = getattr(conf, "stem_weights", [1.0, 1.0])

        # バックボーン
        self.backbone = Backbone(
            sampling_rate=conf.fs,
            num_channels=conf.num_channels,
            n_fft=self.n_fft,
            hop_size=conf.hopSize,
            n_bands=conf.num_bands,
            hidden_size=conf.baseSize,
            num_heads=conf.nHead,
            ffn_hidden_size_factor=conf.hiddenFactor,
            num_layers=conf.nLayers,
            scoring_expansion_factor=conf.scoringExpansionFactor,
            dropout=conf.contextDropoutProb,
            band_split_type=conf.band_split_type,
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
        長尺オーディオを Overlap-Add で分離するユーティリティ。

        Args:
            audio: [T, C] もしくは [C, T]、モノラルなら [T]
            step_in_second: セグメントのホップ長（秒）。未指定時は conf.segmentHopSizeInSecond
            segment_size_in_second: セグメント長（秒）。未指定時は conf.segmentSizeInSecond
            use_hann: True なら重み付けに Hann 窓（※ step < segment のとき推奨）。
            return_numpy: True のとき numpy で返す。

        Returns:
            estimates: [T, N(=2), C]
        """
        device = self.getDevice()

        # ---- 形状の正規化 ----
        x = audio
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not torch.is_floating_point(x):
            x = x.float()
        if x.ndim == 1:
            x = x[:, None]  # [T,1]
        # [C,T] を検知して [T,C] に揃える（C は 1 or 2 を想定）
        if x.shape[0] in (1, 2) and x.shape[1] > x.shape[0]:
            # ただし [T,C] で T < C というケースは稀なのでこの判定で十分
            x = x.transpose(0, 1)
        x = x.to(device)

        if step_in_second is None:
            step_in_second = self.conf.segmentHopSizeInSecond
        if segment_size_in_second is None:
            segment_size_in_second = self.conf.segmentSizeInSecond

        fs = self.fs
        segment_size = int(math.ceil(segment_size_in_second * fs))
        # hop_size の整数倍に丸める（位相整合のため）
        step_size = int(math.ceil(step_in_second * fs / self.hop_size) * self.hop_size)
        pad_samples = max(segment_size - step_size, 0)

        # 時間次元にパディング（両側）
        x_ct = x.transpose(0, 1)  # [C,T]
        if pad_samples > 0:
            x_ct = F.pad(x_ct, (pad_samples, pad_samples))
        total_length = x_ct.shape[-1]

        # 出力バッファ [N,C,T]
        recon = torch.zeros(self.num_stems, self.num_channels, total_length, device=device, dtype=x_ct.dtype)
        win_buf = torch.zeros_like(recon)

        # 窓（step >= segment の場合は矩形にフォールバック）
        use_rect = step_size >= segment_size
        if use_rect or not use_hann:
            window_1d = torch.ones(segment_size, device=device, dtype=x_ct.dtype)
        else:
            window_1d = torch.hann_window(segment_size, device=device, dtype=x_ct.dtype)
        window = window_1d[None, None, :]  # [1,1,L]

        # ---- OLA ループ ----
        for i in range(0, total_length, step_size):
            j = min(i + segment_size, total_length)
            seg_ct = x_ct[:, i:j]  # [C, L]
            cur_len = seg_ct.shape[-1]
            if cur_len < segment_size:
                seg_ct = F.pad(seg_ct, (0, segment_size - cur_len))

            seg_tc = seg_ct.transpose(0, 1).unsqueeze(0)  # [1, T, C]
            est_btnc = self.separate(seg_tc)  # [1, T, N, C]
            est_nct = est_btnc[0].permute(1, 2, 0).contiguous()  # [T, N, C] → [N, C, T]

            seg_out_len = est_nct.shape[-1]
            end_idx = min(i + seg_out_len, total_length)
            valid = end_idx - i
            win = window[..., :seg_out_len]

            recon[..., i:end_idx] += est_nct[..., :valid] * win[..., :valid]
            win_buf[..., i:end_idx] += win[..., :valid]

        # 正規化（0割回避）
        mask = win_buf > 0
        recon[mask] = recon[mask] / win_buf[mask]

        # パディング除去
        if pad_samples > 0:
            recon = recon[..., pad_samples:-pad_samples]

        # [N,C,T] -> [T,N,C]
        out_tnc = recon.permute(2, 0, 1).contiguous()
        if return_numpy:
            return out_tnc.detach().cpu().numpy()
        return out_tnc

    # ---- 学習用ロス計算 ----
    def log_prob(self, audio_slices: torch.Tensor, target_audio: torch.Tensor | None = None):
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
            loss_spec = multi_resolution_stft_loss(
                recon_audio=recon_bcnT,
                target_audio=target_bcnT,
                stem_weights=self.stem_weights,
                hop_divisor=4,
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
