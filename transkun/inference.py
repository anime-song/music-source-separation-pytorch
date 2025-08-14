import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import moduleconf
from .model import TransKun


def read_audio(path: str, sampling_rate: int) -> np.ndarray:
    """
    入力音声を [T, C] の float32 (-1..1) で返す。モノラルは [T, 1] に揃える。
    """
    # librosa.load は mono=False で [C, T]
    y, _ = librosa.load(path, sr=sampling_rate, mono=False)
    if y.ndim == 1:  # [T]
        y = y[:, None]  # [T, 1]
    else:
        y = y.T  # [T, C]
    return y.astype(np.float32, copy=False)


def main():
    import pkg_resources

    default_weight = pkg_resources.resource_filename(__name__, "pretrained/2.0.pt")
    default_conf = pkg_resources.resource_filename(__name__, "pretrained/2.0.conf")

    parser = argparse.ArgumentParser("TransKun MSS Inference")
    parser.add_argument("audioPath", help="path to the input audio file")
    parser.add_argument(
        "outPath",
        help=(
            "path to the output directory (旧: MIDI出力先)。"
            "本MSS版ではステムWAVを書き出すディレクトリとして扱います。"
        ),
    )
    parser.add_argument("--weight", default=default_weight, help="path to the pretrained weight")
    parser.add_argument("--conf", default=default_conf, help="path to the model conf")
    parser.add_argument(
        "--device",
        default="cpu",
        nargs="?",
        help="The device used to run inference (e.g., cpu, cuda:0).",
    )
    parser.add_argument(
        "--segmentHopSize",
        type=float,
        required=False,
        help="Hop size (seconds) for OLA. Default: value defined in model conf.",
    )
    parser.add_argument(
        "--segmentSize",
        type=float,
        required=False,
        help="Segment size (seconds) for OLA. Default: value defined in model conf.",
    )
    parser.add_argument(
        "--use_state_dict",
        action="store_true",
        help="Load checkpoint['state_dict'] instead of checkpoint['best_state_dict'] if set.",
    )
    parser.add_argument(
        "--audio_save_path",
        type=str,
        default=None,
        help="(Optional) Directory to save separated stems. "
        "If not set, uses outPath; if outPath is not a dir, uses input file's parent.",
    )
    parser.add_argument(
        "--stem_names",
        type=str,
        default="piano,other",
        help="Comma-separated stem names (used for filenames). Length should match model.num_stems.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for maximum compatibility.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audioPath)
    out_path_arg = Path(args.outPath)

    # --- Load config & model ---
    conf_manager = moduleconf.parseFromFile(args.conf)
    transkun_cls: TransKun = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    device = torch.device(args.device)
    checkpoint = torch.load(args.weight, map_location=device)

    model: TransKun = transkun_cls(conf=conf).to(device)

    if not args.no_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        except Exception:
            # 環境によっては未対応。失敗時は素直に未コンパイルで進む。
            pass

    # checkpoint の読み込み
    if ("best_state_dict" not in checkpoint) or (checkpoint["best_state_dict"] is None) or args.use_state_dict:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()
    torch.set_grad_enabled(False)

    # --- Read audio ---
    audio_np = read_audio(str(audio_path), sampling_rate=model.fs)  # [T, C]
    audio_t = torch.from_numpy(audio_np).to(device)

    # --- Separate with overlap-add ---
    estimates_tnc = model.separate_overlap(
        audio_t,
        step_in_second=args.segmentHopSize,
        segment_size_in_second=args.segmentSize,
        use_hann=True,
        return_numpy=False,
    )  # [T, N, C]

    # --- Prepare output directory ---
    if args.audio_save_path is not None:
        save_dir = Path(args.audio_save_path)
    else:
        # outPath を優先的にディレクトリとみなす。ファイル名が来た場合も親を使う。
        save_dir = out_path_arg if out_path_arg.suffix == "" else out_path_arg.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    num_stems = estimates_tnc.shape[1]

    # stem名の決定
    stem_names = [name.strip() for name in args.stem_names.split(",") if name.strip()]
    if len(stem_names) != num_stems:
        # 長さが合わなければ自動で stem{i} を作る（必要なら piano/other を上書き）
        stem_names = [f"stem{i}" for i in range(num_stems)]

    # --- Save stems ---
    estimates_np = estimates_tnc.detach().cpu().numpy()  # [T, N, C]
    for stem_idx in range(num_stems):
        stem_audio = estimates_np[:, stem_idx, :]  # [T, C]
        # soundfile は [-1, 1] float を想定。はみ出しを安全にクリップ。
        stem_audio = np.clip(stem_audio, -1.0, 1.0)
        stem_file = save_dir / f"{base_name}_{stem_names[stem_idx]}.wav"
        sf.write(stem_file, stem_audio, conf.fs, subtype="PCM_16")
        print(f"Saved: {stem_file}")

    print("Done.")


if __name__ == "__main__":
    main()
