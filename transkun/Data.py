import math
import numpy as np
from pathlib import Path
import os
import pickle
import torch
import random
from typing import Tuple, Callable, Optional
import pyloudnorm as pyln
import multiprocessing as mp


# ====== 旧ピクル互換用の最小ダミー ======
class Note:
    """
    互換目的のみ。MSSでは未使用。
    旧pickleに含まれる Note を安全に読み出すためのダミー。
    """

    def __init__(self, start: float = 0.0, end: float = 0.0, pitch: int = 0, velocity: int = 64, **kwargs):
        self.start = float(start)
        self.end = float(end)
        self.pitch = int(pitch)
        self.velocity = int(velocity)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self.__dict__)


# ====== I/O: 高速・安全な切り出し ======
def read_audio_slice(audio_path: str, begin_sec: float, end_sec: float, normalize: bool = True):
    """
    指定区間 [begin_sec, end_sec) を読み込み、float32の [-1,1] 正規化で返す。
    begin<0 や 末尾超過はゼロ埋めで安全に処理。常に shape=(T, C) を返す。
    """
    from scipy.io import wavfile

    resolved_path = str(Path(audio_path.replace("\\", os.sep)).expanduser().resolve())
    sample_rate, data = wavfile.read(resolved_path, mmap=True)  # data: (L,) or (L,C), int/float

    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)  # (L,2) に統一

    start_idx = int(math.floor(begin_sec * sample_rate))
    end_idx = int(math.floor(end_sec * sample_rate))
    length = max(end_idx - start_idx, 0)

    # 出力バッファ (常に float32, ゼロ初期化)
    output = np.zeros((length, data.shape[1]), dtype=np.float32)
    if length == 0:
        return output, sample_rate

    # 入力側の有効範囲
    src_begin = max(start_idx, 0)
    src_end = min(end_idx, data.shape[0])
    if src_end <= src_begin:
        return output, sample_rate  # 全ゼロ

    # 出力側の対応スライス
    dst_begin = max(-start_idx, 0)
    dst_end = dst_begin + (src_end - src_begin)

    # 1回のコピーで正規化も済ませる
    source_view = data[src_begin:src_end]
    if normalize and np.issubdtype(source_view.dtype, np.integer):
        scale = float(np.iinfo(source_view.dtype).max)
        output[dst_begin:dst_end] = source_view.astype(np.float32) / scale
    else:
        output[dst_begin:dst_end] = source_view.astype(np.float32, copy=False)

    return output, sample_rate


# ====== 音量正規化 / ミックス ======
def loudness_normalize(wav: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray:
    if wav.size == 0 or not np.any(wav):
        return wav
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(wav)
    if not np.isfinite(loudness):
        return wav
    gain_db = target_lufs - loudness
    return wav * (10 ** (gain_db / 20.0))


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    signal + noise を指定SNRで合成。
    返り値: (mixture, signal_scaled, noise_scaled)
    - 入出力 shape: (T,C) または 1D→自動2D化→戻し
    - 常に float32
    """
    mono_input = (signal.ndim == 1) and (noise.ndim == 1)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if noise.ndim == 1:
        noise = noise[:, np.newaxis]

    length = min(signal.shape[0], noise.shape[0])
    signal = signal[:length].astype(np.float32, copy=False)
    noise = noise[:length].astype(np.float32, copy=False)

    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if signal_power == 0.0 or noise_power == 0.0:
        mixture = signal + noise
    else:
        scale = math.sqrt(signal_power / noise_power / (10.0 ** (snr_db / 10.0)))
        noise = noise * scale
        mixture = signal + noise

    # クリップ回避の等比スケーリング
    peak = float(np.max(np.abs(mixture))) + 1e-12
    if peak > 1.0:
        mixture /= peak
        signal /= peak
        noise /= peak

    if mono_input:
        return mixture[:, 0], signal[:, 0], noise[:, 0]
    return mixture, signal, noise


# ====== SNRカリキュラム ======
def build_anneal_fn(name_or_fn: str | Callable[[float], float]) -> Callable[[float], float]:
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn == "linear":
        return lambda ratio: 1.0 - ratio
    if name_or_fn == "cosine":
        return lambda ratio: (1.0 + math.cos(math.pi * ratio)) / 2.0
    raise ValueError("anneal must be 'linear', 'cosine', or Callable")


def cosine_scale_skewed(progress_ratio: float, gamma: float) -> float:
    progress_ratio = float(np.clip(progress_ratio, 0.0, 1.0))
    return (1.0 + math.cos(math.pi * (progress_ratio ** float(gamma)))) / 2.0


class RandomizedCurriculumSNR:
    """
    center と spread を段階的に変化させつつ、各 step で一様乱数をサンプリング。
    """

    def __init__(
        self,
        min_snr: float,
        max_snr: float,
        max_step: int,
        min_spread: float,
        max_spread: float,
        center_anneal: str | Callable[[float], float] = "linear",
        spread_anneal: str | Callable[[float], float] = "linear",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)
        self.max_step = int(max_step)
        self.min_spread = float(min_spread)
        self.max_spread = float(max_spread)
        self._center_anneal_fn = build_anneal_fn(center_anneal)
        self._spread_anneal_fn = build_anneal_fn(spread_anneal)
        self.rng = rng if rng is not None else random.Random()

    def center_at(self, step: int) -> float:
        ratio = min(step / self.max_step, 1.0)
        scale = self._center_anneal_fn(ratio)
        return self.min_snr + scale * (self.max_snr - self.min_snr)

    def spread_at(self, step: int) -> float:
        ratio = min(step / self.max_step, 1.0)
        scale = self._spread_anneal_fn(ratio)
        return self.min_spread + scale * (self.max_spread - self.min_spread)

    def __call__(self, step: int) -> float:
        center = self.center_at(step)
        spread = self.spread_at(step)
        return center + self.rng.uniform(-spread, +spread)


# ====== データセット ======
class Dataset:
    """
    初期化時に pickle を一周して、必要な軽量情報だけ保持。
    学習ループ中は pickle を再読込しない（高速化 & I/O削減）。
    """

    def __init__(self, dataset_path: str, meta_pickle_path: str):
        self.dataset_path = dataset_path
        self.meta_pickle_path = meta_pickle_path

        self.durations: list[float] = []
        self.audio_filenames: list[str] = []
        self.other_filenames: list[str] = []
        self.other_exists: list[bool] = []

        with open(self.meta_pickle_path, "rb") as fp:
            while True:
                try:
                    sample = pickle.load(fp)
                except EOFError:
                    break

                duration = float(sample["duration"])
                audio_filename = str(sample["audio_filename"])  # required
                other_filename = str(sample.get("other_filename") or "")

                self.durations.append(duration)
                self.audio_filenames.append(audio_filename)
                self.other_filenames.append(other_filename)
                self.other_exists.append(other_filename != "")

        print(f"Found {len(self.durations)} pieces in {os.path.basename(self.meta_pickle_path)}")
        print("totalDuration:", sum(self.durations))

    def __getstate__(self):
        # マルチプロセスでのシリアライズ対応
        return {
            "dataset_path": self.dataset_path,
            "meta_pickle_path": self.meta_pickle_path,
        }

    def __setstate__(self, state):
        self.__init__(state["dataset_path"], state["meta_pickle_path"])

    def fetch_data(
        self,
        idx: int,
        begin_sec: float,
        end_sec: float,
        audio_normalize: bool,
        other_idx: Optional[int] = None,
        other_begin: Optional[float] = None,
        other_end: Optional[float] = None,
    ):
        audio_filename = self.audio_filenames[idx]

        # other の選択：指定がなければ同一曲、指定があれば別曲
        if other_idx is None:
            other_filename = self.other_filenames[idx]
        else:
            other_filename = self.other_filenames[other_idx]

        if not other_filename:
            raise RuntimeError("other_filename が空です。呼び出し側で補完してください。")

        if other_begin is None or other_end is None:
            other_begin, other_end = begin_sec, end_sec

        # 切り出し
        audio_path = os.path.join(self.dataset_path, audio_filename)
        other_path = os.path.join(self.dataset_path, other_filename)

        target_audio, sample_rate = read_audio_slice(audio_path, begin_sec, end_sec, audio_normalize)
        other_slice, _ = read_audio_slice(other_path, other_begin, other_end, audio_normalize)

        return target_audio, other_slice, sample_rate


class AugmentatorAudiomentations:
    def __init__(
        self,
        sample_rate: int = 44100,
        pitch_shift_range=(-0.2, 0.2),
        eq_db_range=(-3, 3),
        snr_range=(0, 40),
        conv_ir_folder: Optional[str] = None,
        noise_folder: Optional[str] = None,
    ):
        from audiomentations import (
            AddGaussianSNR,
            Compose,
            PitchShift,
            ApplyImpulseResponse,
            AddBackgroundNoise,
            SevenBandParametricEQ,
            PolarityInversion,
            RoomSimulator,
        )

        self.sample_rate = sample_rate

        musical_chain = [
            PitchShift(*pitch_shift_range, p=0.5),
            SevenBandParametricEQ(*eq_db_range, p=0.5),
            PolarityInversion(p=0.5),
            RoomSimulator(calculation_mode="rt60", max_order=3, p=0.5),
        ]
        self.transform = Compose(musical_chain)

        self.reverb = None
        if conv_ir_folder is not None:
            ir_files = list(Path(conv_ir_folder).glob("**/*.wav"))
            if ir_files:
                self.reverb = ApplyImpulseResponse(ir_files, p=0.5, lru_cache_size=2000, leave_length_unchanged=True)
                print("aug: convIR enabled")

        noise_chain = []
        if noise_folder is not None:
            noise_files = list(Path(noise_folder).glob("**/*.wav"))
            if noise_files:
                noise_sub = Compose([PolarityInversion(), PitchShift(), SevenBandParametricEQ(*eq_db_range, p=0.5)])
                noise_chain.append(
                    AddBackgroundNoise(noise_files, *snr_range, p=0.7, lru_cache_size=256, noise_transform=noise_sub)
                )
                print("aug: noise enabled")
        noise_chain.append(AddGaussianSNR(min_snr_db=snr_range[0], max_snr_db=snr_range[1], p=0.1))
        self.transform_noise = Compose(noise_chain)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32, order="C").T  # (C,T)
        x = self.transform(x, sample_rate=self.sample_rate)
        if self.reverb is not None:
            x_reverb = self.reverb(x, sample_rate=self.sample_rate)
            wet_ratio = random.random()
            x = wet_ratio * x + (1.0 - wet_ratio) * x_reverb
        x = self.transform_noise(x, sample_rate=self.sample_rate)
        return x.T  # (T,C)


# ====== イテレータ ======
class DatasetIterator(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        hop_size_in_second: float,
        chunk_size_in_second: float,
        step_counter: Optional[mp.Value] = None,
        snr_scheduler: Optional[Callable[[int], float]] = None,
        audio_normalize: bool = True,
        dithering_frames: bool = True,
        seed: int = 1234,
        augmentator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        cross_mix_prob: float = 0.25,
    ):
        """
        cross_mix_prob: 同一曲の other ではなく、別曲の other を使う確率。
        legacy_kwargs: 旧シグネチャ (hopSizeInSecond, chunkSizeInSecond, audioNormalize, ditheringFrames など) を許容。
        """
        super().__init__()
        self.dataset = dataset
        self.hop_size_in_second = float(hop_size_in_second)
        self.chunk_size_in_second = float(chunk_size_in_second)
        self.step_counter = step_counter
        self.snr_scheduler = snr_scheduler
        self.audio_normalize = audio_normalize
        self.dithering_frames = dithering_frames
        self.augmentator = augmentator
        self.cross_mix_prob = float(cross_mix_prob)

        random_generator = random.Random(seed)

        # チャンク列を事前生成
        chunks_all = []
        for piece_index, duration in enumerate(self.dataset.durations):
            duration = float(duration)
            hop = self.hop_size_in_second
            size = self.chunk_size_in_second
            num_chunks = math.ceil((duration + size) / hop)
            hop_per_chunk = math.ceil(size / hop)

            for j in range(-hop_per_chunk, num_chunks + hop_per_chunk):
                shift = (random_generator.random() - 0.5) if self.dithering_frames else 0.0
                begin = (j + shift) * hop - size / 2.0
                end = begin + size
                if (begin < duration) and (end > 0.0):
                    chunks_all.append((piece_index, begin, end, self.dataset.other_exists[piece_index]))

        random_generator.shuffle(chunks_all)
        self.chunks_all = chunks_all
        self.chunks_with_other = [c for c in chunks_all if c[3]]

    def __len__(self) -> int:
        return len(self.chunks_all)

    def _current_snr_db(self) -> float:
        if self.snr_scheduler is None or self.step_counter is None:
            return 0.0
        with self.step_counter.get_lock():
            step = int(self.step_counter.value)
        return float(self.snr_scheduler(step))

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError

        piece_index, begin, end, has_other = self.chunks_all[index]

        # 別曲からのクロスミックスを使うか？
        use_cross = (random.random() < self.cross_mix_prob) or (not has_other)
        if use_cross:
            if not self.chunks_with_other:
                raise RuntimeError("other_exists==True のチャンクが 1 つもありません")
            other_piece_index, other_begin, other_end, _ = random.choice(self.chunks_with_other)
        else:
            other_piece_index, other_begin, other_end = None, None, None

        target_audio, other_slice, sample_rate = self.dataset.fetch_data(
            idx=piece_index,
            begin_sec=begin,
            end_sec=end,
            audio_normalize=self.audio_normalize,
            other_idx=other_piece_index,
            other_begin=other_begin,
            other_end=other_end,
        )

        # Augmentation は target のみ（必要なら other にも適用してOK）
        if self.augmentator is not None:
            target_audio = self.augmentator(target_audio)

        # ラウドネス & ミックス
        snr_db = self._current_snr_db()
        target_audio = loudness_normalize(target_audio, sample_rate)
        other_slice = loudness_normalize(other_slice, sample_rate)
        mixture, target_scaled, other_scaled = mix_at_snr(target_audio, other_slice, snr_db)

        # 形状を常に [T, 2, C] に統一
        if target_scaled.ndim == 1:
            target_scaled = target_scaled[:, np.newaxis]
        if other_scaled.ndim == 1:
            other_scaled = other_scaled[:, np.newaxis]
        stacked_targets = np.stack([target_scaled, other_scaled], axis=1)  # (T,2,C)

        # snake_case を正としつつ、既存コード互換の別名も同梱
        sample = {
            "audio_slices": mixture.astype(np.float32, copy=False),  # (T,C)
            "target_audio": stacked_targets.astype(np.float32, copy=False),  # (T,2,C)
            "sample_rate": sample_rate,
            "begin": float(begin),
        }
        # 互換キー（旧コードは audioSlices / fs を参照）
        sample["audioSlices"] = sample["audio_slices"]
        sample["fs"] = sample["sample_rate"]
        return sample


# ====== Collate ======
def collate_fn(batch):
    return batch  # デバッグ用：バッチ化しない


def collate_fn_batching(batch):
    """
    - audio_slices: (B,T,C)
    - target_audio: (B,T,2,C)

    snake_case("audio_slices") を優先しつつ、旧キー("audioSlices") もフォールバックで受理。
    """

    def _get_audio_slices(sample):
        if "audio_slices" in sample:
            return torch.as_tensor(sample["audio_slices"], dtype=torch.float32)
        return torch.as_tensor(sample["audioSlices"], dtype=torch.float32)

    audio_slices_list = [_get_audio_slices(s) for s in batch]
    targets_list = [torch.as_tensor(s["target_audio"], dtype=torch.float32) for s in batch]

    # 長さを最小に揃える（ずれは1サンプルまで許容される想定）
    min_len = min(x.shape[0] for x in audio_slices_list)
    max_len = max(x.shape[0] for x in audio_slices_list)
    assert max_len - min_len < 2, "サンプル長が揃っていません"

    audio_slices_list = [x[:min_len] for x in audio_slices_list]
    targets_list = [x[:min_len] for x in targets_list]

    audio_slices = torch.stack(audio_slices_list, dim=0)  # (B,T,C)
    targets = torch.stack(targets_list, dim=0)  # (B,T,2,C)

    return {"audioSlices": audio_slices, "target_audio": targets}
