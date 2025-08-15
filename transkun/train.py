import os
import random

from torch.utils.tensorboard import SummaryWriter
import torch
from .model import TransKun
from .Data import (
    Dataset,
    DatasetIterator,
    collate_fn_batching,
    AugmentatorAudiomentations,
)
import copy
import time
import numpy as np
import math
from pathlib import Path

from .train_utils import *
import argparse

import moduleconf


def check_gradients(model) -> None:
    """
    勾配の健全性チェック（必要に応じて有効化）。
    """
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    assert len(params_with_grad) == sum(1 for _ in model.parameters()), "一部 grad が None"
    assert all(torch.isfinite(p.grad).all() for p in params_with_grad), "grad に Inf / NaN"
    print(f"✓gradients OK")


import torch.nn.functional as F


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    """
    [B, T, C] or [B, T, 2, C] を [B, -1] にフラット化（バッチごと）。
    """
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SNR[dB] をバッチごとに計算して [B] を返す。
    10 * log10(||s||^2 / ||s - ŝ||^2)
    """
    ref = _flatten_bt(reference)
    est = _flatten_bt(estimate)
    noise = ref - est
    num = (ref**2).sum(dim=1)
    den = (noise**2).sum(dim=1) + eps
    return 10.0 * torch.log10((num + eps) / den)


@torch.no_grad()
def si_snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SNR[dB] をバッチごとに計算して [B] を返す。
    参照・推定ともゼロ平均化 → 射影成分と残差で定義。
    """
    ref = _flatten_bt(reference)
    est = _flatten_bt(estimate)

    ref = ref - ref.mean(dim=1, keepdim=True)
    est = est - est.mean(dim=1, keepdim=True)

    ref_power = (ref**2).sum(dim=1, keepdim=True) + eps
    proj = (torch.sum(est * ref, dim=1, keepdim=True) / ref_power) * ref  # s_target
    noise = est - proj

    num = (proj**2).sum(dim=1)
    den = (noise**2).sum(dim=1) + eps
    return 10.0 * torch.log10((num + eps) / den)


@torch.no_grad()
def _separate_stems(model: TransKun, audio_slices: torch.Tensor) -> torch.Tensor | None:
    """
    可能なら分離推定 [B, T, 2, C] を取得。なければ None を返す。
    優先: model.separate → model(...)。
    """
    preds = None
    if hasattr(model, "separate"):
        preds = model.separate(audio_slices)
    else:
        try:
            preds = model(audio_slices)
        except Exception:
            preds = None

    if isinstance(preds, (tuple, list)):  # 返り値が複合のときは最初の要素を試す
        preds = preds[0]
    if preds is None:
        return None
    if preds.ndim != 4 or preds.shape[2] != 2:
        # 期待形状 [B, T, 2, C] でない場合は非対応として None
        return None
    return preds


def validate_mss(
    model: TransKun,
    dataloader_val: torch.utils.data.DataLoader,
    device: torch.device,
    loss_spec_weight: float,
) -> dict[str, float]:
    """
    MSS バリデーション：
      - 再構成ロス（spec + wmse * weight）
      - 可能なら SNR / SI-SNR をステム別に集計
    いずれも音声長(秒)で重み付け平均。
    """
    model.eval()

    total_weight = 0.0
    weighted_loss_sum = 0.0
    loss_spec_sum = 0.0
    loss_wmse_sum = 0.0
    num_batches = 0

    # 指標用（推定が取れた場合のみ加算）
    metrics_weighted = {
        "snr_target_db": 0.0,
        "snr_other_db": 0.0,
        "si_snr_target_db": 0.0,
        "si_snr_other_db": 0.0,
    }
    metrics_available = False  # 少なくとも1バッチでも推定に成功したら True

    with torch.no_grad():
        for batch in dataloader_val:
            audio_slices = batch["audioSlices"].to(device)  # [B, T, C]
            target_audio = batch["target_audio"].to(device)  # [B, T, 2, C]
            batch_seconds = audio_slices.shape[1] / model.conf.fs
            batch_weight = float(batch_seconds)

            # --- ロス ---
            loss_spec, loss_wmse = model.calc_loss(audio_slices, target_audio=target_audio)
            loss_recon = (loss_spec + loss_wmse * model.loss_wmse_weight) * loss_spec_weight
            total_loss = loss_recon

            weighted_loss_sum += float(total_loss.item()) * batch_weight
            total_weight += batch_weight
            loss_spec_sum += float(loss_spec.item())
            loss_wmse_sum += float(loss_wmse.item())
            num_batches += 1

            # --- 分離推定が取れれば SNR / SI-SNR ---
            estimates = _separate_stems(model, audio_slices)
            if estimates is not None:
                # target=stem0, other=stem1
                est_target = estimates[:, :, 0, :]  # [B,T,C]
                est_other = estimates[:, :, 1, :]

                tgt_target = target_audio[:, :, 0, :]
                tgt_other = target_audio[:, :, 1, :]

                snr_target = snr_db(tgt_target, est_target).mean().item()
                snr_other = snr_db(tgt_other, est_other).mean().item()
                si_target = si_snr_db(tgt_target, est_target).mean().item()
                si_other = si_snr_db(tgt_other, est_other).mean().item()

                metrics_weighted["snr_target_db"] += snr_target * batch_weight
                metrics_weighted["snr_other_db"] += snr_other * batch_weight
                metrics_weighted["si_snr_target_db"] += si_target * batch_weight
                metrics_weighted["si_snr_other_db"] += si_other * batch_weight
                metrics_available = True

    mean_weighted_loss = weighted_loss_sum / max(total_weight, 1e-8)
    result = {
        "val_loss": mean_weighted_loss,
        "val_loss_spec": loss_spec_sum / max(num_batches, 1),
        "val_loss_wmse": loss_wmse_sum / max(num_batches, 1),
    }

    if metrics_available:
        for k, v in metrics_weighted.items():
            result[f"val_{k}"] = v / max(total_weight, 1e-8)

    return result


def train(worker_id: int, filename: str, run_seed: int, args):
    device = torch.device("cuda:" + str(worker_id % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    random.seed(worker_id + int(time.time()))
    np.random.seed(worker_id + int(time.time()))
    torch.manual_seed(worker_id + int(time.time()))
    torch.cuda.manual_seed(worker_id + int(time.time()))

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # モデル設定の取得
    conf_manager = moduleconf.parseFromFile(args.modelConf)
    transkun_model = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config
    model: TransKun

    if worker_id == 0:
        # 既存チェックポイントがなければ初期化
        if not os.path.exists(filename):
            print("initializing the model...")

            (
                start_epoch,
                start_iter,
                model,
                loss_tracker,
                best_state_dict,
                optimizer,
                lr_scheduler,
            ) = initializeCheckpoint(
                transkun_model,
                device=device,
                max_lr=args.max_lr,
                weight_decay=args.weight_decay,
                nIter=args.nIter,
                conf=conf,
            )

            save_checkpoint(
                filename,
                start_epoch,
                start_iter,
                model,
                loss_tracker,
                best_state_dict,
                optimizer,
                lr_scheduler,
            )

    (
        start_epoch,
        start_iter,
        model,
        loss_tracker,
        best_state_dict,
        optimizer,
        lr_scheduler,
    ) = load_checkpoint(transkun_model, conf, filename, device)
    print(f"#{worker_id} loaded")

    if worker_id == 0:
        print("loading dataset....")

    dataset_path = args.datasetPath
    dataset_meta_train = args.datasetMetaFile_train
    dataset_meta_val = args.datasetMetaFile_val
    snr_values_path = args.snr_values_path

    dataset = Dataset(dataset_path, dataset_meta_train)
    dataset_val = Dataset(dataset_path, dataset_meta_val)

    print(f"#{worker_id} loaded")

    if worker_id == 0:
        writer = SummaryWriter(filename + ".log")

    global_step_int = start_iter
    # ハイパーパラメータ
    batch_size = args.batch_size
    loss_spec_weight = conf.loss_spec_weight

    hop_size = conf.segmentHopSizeInSecond if args.hopSize is None else args.hopSize
    chunk_size = conf.segmentSizeInSecond if args.chunkSize is None else args.chunkSize

    grad_norm_history = MovingBuffer(initValue=40, maxLen=10000)

    augmentator = None
    if args.augment:
        augmentator = AugmentatorAudiomentations(
            sample_rate=44100, noise_folder=args.noiseFolder, conv_ir_folder=args.irFolder
        )

    for epoch in range(start_epoch, 1000000):
        # イテレータ／ローダの組み立て
        data_iter = DatasetIterator(
            dataset,
            hop_size,
            chunk_size,
            seed=epoch * 100 + run_seed,
            augmentator=augmentator,
            snr_values_path=snr_values_path,
        )

        dataloader = torch.utils.data.DataLoader(
            data_iter,
            batch_size=batch_size,
            collate_fn=collate_fn_batching,
            num_workers=args.dataLoaderWorkers,
            shuffle=True,
            drop_last=True,
            prefetch_factor=max(4, args.dataLoaderWorkers),
        )

        # ログ用集計
        train_loss_weighted_sum = 0.0
        train_duration_sum = 0.0

        global_step_warmup_cutoff = global_step_int + 500

        for batch_index, batch in enumerate(dataloader):
            if worker_id == 0:
                current_lr = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar("Optimizer/lr", current_lr, global_step_int)

            t1 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # 入力
            audio_slices = batch["audioSlices"].to(device)
            target_audio = batch["target_audio"].to(device)

            # ロス計算
            loss_spec, loss_wmse = model.calc_loss(audio_slices, target_audio=target_audio)
            loss_recon = (loss_spec + loss_wmse * model.loss_wmse_weight) * loss_spec_weight
            total_loss = loss_recon

            total_loss.backward()

            # 勾配クリッピング（適応）
            current_clip_value = grad_norm_history.getQuantile(args.gradClippingQuantile)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip_value)
            grad_norm_history.step(float(total_grad_norm.item()))

            optimizer.step()

            try:
                if global_step_int > global_step_warmup_cutoff:
                    lr_scheduler.step()
            except Exception:
                # 既定の最終イテレーション後などは例外が出る可能性あり
                pass

            # ログ用に時間重み付きで平均化
            audio_length_sec = audio_slices.shape[1] / model.conf.fs
            train_loss_weighted_sum += float(total_loss.item()) * float(audio_length_sec)
            train_duration_sum += float(audio_length_sec)

            if worker_id == 0:
                t2 = time.time()
                mean_train_loss = train_loss_weighted_sum / max(train_duration_sum, 1e-8)

                print(
                    "epoch:{:d} progress:{:0.3f} step:{}  "
                    "mean_train_loss:{:0.4f} loss_spec:{:0.4f} loss_wmse:{:0.4f} "
                    "gradNorm:{:0.2f} clipValue:{:0.2f} time:{:0.2f}".format(
                        epoch,
                        batch_index / len(dataloader),
                        global_step_int,
                        mean_train_loss,
                        loss_spec.item(),
                        loss_wmse.item(),
                        total_grad_norm.item(),
                        current_clip_value,
                        t2 - t1,
                    )
                )
                writer.add_scalar("Loss/train_mean", mean_train_loss, global_step_int)
                writer.add_scalar("Loss/train_loss_spec", loss_spec.item(), global_step_int)
                writer.add_scalar("Loss/train_loss_wmse", loss_wmse.item(), global_step_int)
                writer.add_scalar("Optimizer/gradNorm", total_grad_norm.item(), global_step_int)
                writer.add_scalar("Optimizer/clipValue", current_clip_value, global_step_int)

                compute_train_metrics = (args.trainMetricInterval > 0) and (batch_index % args.trainMetricInterval == 0)
                if worker_id == 0 and compute_train_metrics:
                    with torch.no_grad():
                        model.eval()
                        # 推定 [B, T, 2, C]
                        estimates = model.separate(audio_slices)

                        # 教師 [B, T, 2, C] からステムを取り出す
                        tgt_target = target_audio[:, :, 0, :]  # [B, T, C]
                        tgt_other = target_audio[:, :, 1, :]

                        est_target = estimates[:, :, 0, :]
                        est_other = estimates[:, :, 1, :]

                        snr_target_values = snr_db(tgt_target, est_target)  # [B]
                        snr_other_values = snr_db(tgt_other, est_other)  # [B]

                        # バッチ平均
                        snr_target_mean = snr_target_values.mean().item()
                        snr_other_mean = snr_other_values.mean().item()

                    # TensorBoard ログ
                    writer.add_scalar("Metrics/train_snr_target_db", snr_target_mean, global_step_int)
                    writer.add_scalar("Metrics/train_snr_other_db", snr_other_mean, global_step_int)

                    # コンソール出力にも追加
                    print(
                        "    [Train Metrics] SNR target:{:.2f} dB | other:{:.2f} dB".format(
                            snr_target_mean, snr_other_mean
                        )
                    )

                if math.isnan(mean_train_loss):
                    raise RuntimeError("Loss is NaN")

                # 周期的にチェックポイント保存
                if batch_index % 2000 == 1999:
                    save_checkpoint(
                        filename,
                        epoch + 1,
                        global_step_int + 1,
                        model,
                        loss_tracker,
                        best_state_dict,
                        optimizer,
                        lr_scheduler,
                    )
                    print("saved")

            global_step_int += 1

        if worker_id == 0:
            print("Validating...")

        # ===== Validation (MSS only) =====
        torch.cuda.empty_cache()

        data_iter_val = DatasetIterator(
            dataset_val,
            hopSizeInSecond=conf.segmentHopSizeInSecond,
            chunkSizeInSecond=chunk_size,
            seed=run_seed + epoch * 100,
            snr_values_path=snr_values_path,
        )
        dataloader_val = torch.utils.data.DataLoader(
            data_iter_val,
            batch_size=2 * batch_size,
            collate_fn=collate_fn_batching,  # そのまま利用
            num_workers=args.dataLoaderWorkers,
            shuffle=True,
        )

        val_result = validate_mss(model, dataloader_val, device, loss_spec_weight=loss_spec_weight)

        torch.cuda.empty_cache()

        if worker_id == 0:
            # ログとベスト更新（最小の val_loss を採用）
            mean_train_loss_epoch = train_loss_weighted_sum / max(train_duration_sum, 1e-8)
            loss_tracker["train"].append(mean_train_loss_epoch)
            loss_tracker["val"].append(val_result["val_loss"])

            print("val_result:", val_result)
            for key, value in val_result.items():
                writer.add_scalar("val/" + key, value, epoch)

            is_best = (len(loss_tracker["val"]) == 0) or (val_result["val_loss"] <= min(loss_tracker["val"]))
            if is_best:
                print("best updated")
                best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                filename,
                epoch + 1,
                global_step_int + 1,
                model,
                loss_tracker,
                best_state_dict,
                optimizer,
                lr_scheduler,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Training (MSS only)")
    parser.add_argument("saved_filename")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--datasetPath", required=True)
    parser.add_argument("--datasetMetaFile_train", required=True)
    parser.add_argument("--datasetMetaFile_val", required=True)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--hopSize", required=False, type=float)
    parser.add_argument("--chunkSize", required=False, type=float)
    parser.add_argument("--dataLoaderWorkers", default=2, type=int)
    parser.add_argument("--gradClippingQuantile", default=0.8, type=float)

    parser.add_argument("--max_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nIter", default=500000, type=int)
    parser.add_argument("--modelConf", required=True, help="the path to the model conf file")
    parser.add_argument("--augment", action="store_true", help="do data augmentation")
    parser.add_argument("--noiseFolder", required=False)
    parser.add_argument("--irFolder", required=False)
    parser.add_argument(
        "--trainMetricInterval",
        default=100,
        type=int,
        help="training中にSNRを算出してログする間隔（ステップ）。0で無効。",
    )
    parser.add_argument("--snr_values_path", type=Path, default=None)

    args = parser.parse_args()
    saved_filename = args.saved_filename

    run_seed = int(time.time())

    train(0, saved_filename, run_seed, args)
