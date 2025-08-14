from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{num_bytes} B"


def strip_checkpoint_dict(
    ckpt: Dict[str, Any],
    remove_optimizer: bool = True,
    remove_scheduler: bool = True,
    keep_only_best: bool = False,
) -> Dict[str, Any]:
    """Return a new dict with requested fields removed/adjusted."""
    out = dict(ckpt)  # shallow copy

    # Optionally collapse to best_state_dict
    if keep_only_best:
        best = out.get("best_state_dict", None)
        if best is not None:
            out["state_dict"] = best
        # keep best_state_dict as-is (useful for downstream),
        # or drop it if you really want minimal file:
        # del out["best_state_dict"]

    if remove_optimizer and "optimizer_state_dict" in out:
        del out["optimizer_state_dict"]

    if remove_scheduler and "lr_scheduler_state_dict" in out:
        del out["lr_scheduler_state_dict"]

    return out


def save_checkpoint_stripped(
    in_path: Path,
    out_path: Path,
    remove_optimizer: bool = True,
    remove_scheduler: bool = True,
    keep_only_best: bool = False,
) -> None:
    in_path = in_path.resolve()
    out_path = out_path.resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    before_bytes = in_path.stat().st_size
    print(f"[INFO] Loading: {in_path} ({human_size(before_bytes)})")

    # Load on CPU to avoid GPU RAM usage
    ckpt = torch.load(str(in_path), map_location="cpu")

    # Strip
    ckpt_stripped = strip_checkpoint_dict(
        ckpt,
        remove_optimizer=remove_optimizer,
        remove_scheduler=remove_scheduler,
        keep_only_best=keep_only_best,
    )

    # Save (zipfile serialization is default in modern PyTorch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_stripped, str(out_path))

    after_bytes = out_path.stat().st_size
    delta = before_bytes - after_bytes
    ratio = (after_bytes / before_bytes) if before_bytes > 0 else 0.0
    print(
        f"[DONE] Saved: {out_path} ({human_size(after_bytes)})  "
        f"[reduced {human_size(delta)} | {ratio:.2%} of original]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip optimizer/scheduler from a PyTorch checkpoint")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input checkpoint path(s). You can pass multiple files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file or directory. "
        "If multiple inputs are given, this must be a directory. "
        "Default: write alongside input with suffix '.stripped.pt'.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the input file in place (creates a temporary file then moves it).",
    )
    parser.add_argument(
        "--keep_only_best",
        action="store_true",
        help="Set state_dict = best_state_dict if available (keeps both keys).",
    )
    parser.add_argument(
        "--keep_optimizer",
        action="store_true",
        help="Do NOT remove optimizer_state_dict (default is to remove).",
    )
    parser.add_argument(
        "--keep_scheduler",
        action="store_true",
        help="Do NOT remove lr_scheduler_state_dict (default is to remove).",
    )

    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    out_arg = Path(args.output) if args.output else None

    # Validate output target
    if len(paths) > 1 and out_arg is not None and (out_arg.exists() and out_arg.is_file()):
        print("[ERROR] When passing multiple inputs, --output must be a directory.", file=sys.stderr)
        sys.exit(2)

    for in_path in paths:
        if args.inplace:
            tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")
            save_checkpoint_stripped(
                in_path=in_path,
                out_path=tmp_path,
                remove_optimizer=not args.keep_optimizer,
                remove_scheduler=not args.keep_scheduler,
                keep_only_best=args.keep_only_best,
            )
            # atomic-ish replace
            tmp_path.replace(in_path)
            print(f"[INFO] Replaced original in-place: {in_path}")
        else:
            if out_arg is None:
                out_path = in_path.with_name(in_path.stem + ".stripped" + in_path.suffix)
            else:
                out_path = out_arg
                if out_path.exists() and out_path.is_dir():
                    out_path = out_path / (in_path.stem + ".stripped" + in_path.suffix)
                elif len(paths) > 1:
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path = out_path / (in_path.stem + ".stripped" + in_path.suffix)

            save_checkpoint_stripped(
                in_path=in_path,
                out_path=out_path,
                remove_optimizer=not args.keep_optimizer,
                remove_scheduler=not args.keep_scheduler,
                keep_only_best=args.keep_only_best,
            )


if __name__ == "__main__":
    main()
