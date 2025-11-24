"""Standalone training/evaluation pipeline for the image-only recommendation tower.

This script mirrors the behavior of `debug_text_model.py`, but focuses on the
image features to keep the original code untouched.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import project_config


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class MovieDataset(Dataset):
    """Dataset that keeps both text and image towers to reuse infrastructure."""

    def __init__(self, csv_path: str, text_feature_path: str, img_feature_path: str):
        super().__init__()
        self.data_df = pd.read_csv(csv_path, sep=",", engine="python")

        self.text_ids, self.text_feature_array = self._load_npz_arrays(text_feature_path)
        self.img_ids, self.img_feature_array = self._load_npz_arrays(img_feature_path)

        self.text_feature_dict = self._construct_feature_dict(self.text_ids, self.text_feature_array)
        self.img_feature_dict = self._construct_feature_dict(self.img_ids, self.img_feature_array)
        self.all_movie_idx_map = self._check_missing_features_get_idx_map()
        self.user_id_map = self._build_user_id_dict()

        self.text_feature_tensor = torch.from_numpy(self.text_feature_array).float()
        self.img_feature_tensor = torch.from_numpy(self.img_feature_array).float()

    def __len__(self) -> int:
        return len(self.data_df)

    @staticmethod
    def _load_npz_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=True) as npz_data:
            ids = npz_data["ids"].copy()
            features = npz_data["features"].copy()
        return ids, features

    def _construct_feature_dict(self, ids: np.ndarray, features: np.ndarray) -> Dict[int, np.ndarray]:
        feature_dict: Dict[int, np.ndarray] = {}
        for idx, movie_id in enumerate(ids):
            if movie_id in feature_dict:
                raise ValueError(f"AustinERROR duplicate movie id found: {movie_id}")
            feature_dict[movie_id] = features[idx]
        return feature_dict

    def _check_missing_features_get_idx_map(self) -> Dict[int, int]:
        text_keys = list(self.text_feature_dict.keys())
        img_keys = list(self.img_feature_dict.keys())
        if set(text_keys) != set(img_keys):
            missing_in_text = set(img_keys) - set(text_keys)
            missing_in_img = set(text_keys) - set(img_keys)
            raise ValueError(
                "AustinERROR Missing features found. "
                f"Missing in text: {missing_in_text}, Missing in img: {missing_in_img}"
            )
        movie_idx_map = {}
        for idx, movie_id in enumerate(text_keys):
            movie_idx_map[movie_id] = idx
        return movie_idx_map

    def _build_user_id_dict(self) -> Dict[int, int]:
        user_ids = self.data_df["user_id"].unique().tolist()
        return {user_id: idx for idx, user_id in enumerate(user_ids)}

    def construct_positive_dict(self) -> Dict[int, set[int]]:
        positive_dict: Dict[int, set[int]] = {}
        for row in self.data_df.itertuples():
            if row.label == 1:
                positive_dict.setdefault(row.user_id, set()).add(row.movie_id)
        return positive_dict

    def get_interact_dict(self, sample_num: int = 100) -> Dict[int, Dict[str, List[int]]]:
        interact_dict = self.data_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        all_movies = set(self.all_movie_idx_map.keys())
        rng = random.Random(42)
        for user_id, interacted_movies in interact_dict.items():
            non_interacted = list(all_movies - interacted_movies)
            if not non_interacted:
                interact_dict[user_id] = {
                    "interact_movies": interacted_movies,
                    "non_interact_movies": [],
                }
                continue
            sample_count = min(sample_num, len(non_interacted))
            rng.shuffle(non_interacted)
            interact_dict[user_id] = {
                "interact_movies": interacted_movies,
                "non_interact_movies": non_interacted[:sample_count],
            }
        return interact_dict

    def get_img_tensor_for_movie_ids(
        self, movie_ids: Sequence[int], device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        idxs = []
        for movie_id in movie_ids:
            idx = self.all_movie_idx_map.get(movie_id)
            if idx is not None:
                idxs.append(idx)
        if not idxs:
            return None
        feats = self.img_feature_tensor[idxs]
        if device is not None:
            feats = feats.to(device, non_blocking=True)
        return feats

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data_df.iloc[index]
        sample = {
            "movie_id": torch.tensor(row["movie_id"], dtype=torch.long),
            "user_id": torch.tensor(row["user_id"], dtype=torch.long),
            "rating": torch.tensor(row["rating"], dtype=torch.float),
            "label": torch.tensor(row["label"], dtype=torch.float),
            "text_features": torch.tensor(self.text_feature_dict[row["movie_id"]], dtype=torch.float),
            "img_features": torch.tensor(self.img_feature_dict[row["movie_id"]], dtype=torch.float),
        }
        return sample


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class ImageModel(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, num_users: int, dropout_rate: float, user_embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.user_proj = nn.Linear(user_embedding_dim, proj_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, user_id_indices: torch.Tensor, img_features: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_id_indices)
        u = self.user_proj(u)
        v = self.fc(img_features)
        score = torch.sum(u * v, dim=1)
        return score


# ---------------------------------------------------------------------------
# TRAIN / EVAL HELPERS
# ---------------------------------------------------------------------------
def _map_user_ids_to_indices(user_ids: torch.Tensor, user_id_map: Dict[int, int], device: torch.device) -> torch.Tensor:
    indices = []
    for uid in user_ids:
        uid_int = int(uid.item())
        if uid_int not in user_id_map:
            raise ValueError(f"AustinERROR Unknown user_id {uid_int} encountered during training.")
        indices.append(user_id_map[uid_int])
    return torch.tensor(indices, dtype=torch.long, device=device)


def _safe_roc_auc(y_true: List[float], y_pred: List[float], logger: logging.Logger) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError as exc:
        logger.warning(f"AUC calculation failed ({exc}); returning 0.5 as fallback.")
        return 0.5


def _setup_logger(model_name: str) -> logging.Logger:
    logs_dir = pathlib.Path("logs") / model_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_{timestamp}.log"

    logger = logging.getLogger(f"trainer.{model_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info(f"Logging to: {log_path}")
    return logger


@dataclass
class TopKSettings:
    ks: Sequence[int]
    sample_neg: int
    max_users: Optional[int]


def compute_topk_metrics(
    model: ImageModel,
    val_ds: MovieDataset,
    all_ds: MovieDataset,
    topk_settings: TopKSettings,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    positive_dict = val_ds.construct_positive_dict()
    neg_dict = all_ds.get_interact_dict(sample_num=topk_settings.sample_neg)

    ks = sorted(set(topk_settings.ks))
    precision_k = {k: [] for k in ks}
    recall_k = {k: [] for k in ks}
    ndcg_k = {k: [] for k in ks}
    hit_k = {k: [] for k in ks}
    map_list: List[float] = []
    mrr_list: List[float] = []

    user_ids = list(positive_dict.keys())
    if topk_settings.max_users and len(user_ids) > topk_settings.max_users:
        rng = random.Random(42)
        user_ids = rng.sample(user_ids, topk_settings.max_users)

    with torch.no_grad():
        for user_id in tqdm.tqdm(user_ids, desc="Computing Top-K metrics"):
            pos_items = positive_dict.get(user_id, set())
            if not pos_items:
                continue
            neg_info = neg_dict.get(user_id)
            if not neg_info:
                continue
            candidate_items = list(pos_items) + neg_info["non_interact_movies"]
            if not candidate_items:
                continue

            img_feature_tensor = all_ds.get_img_tensor_for_movie_ids(candidate_items, device=device)
            if img_feature_tensor is None or img_feature_tensor.size(0) == 0:
                if logger:
                    logger.debug(f"Skip user {user_id}: unable to fetch image features.")
                continue

            user_idx = all_ds.user_id_map.get(user_id)
            if user_idx is None:
                continue
            user_tensor = torch.full((img_feature_tensor.size(0),), user_idx, dtype=torch.long, device=device)

            scores = model(user_tensor, img_feature_tensor).squeeze(-1)
            sorted_scores, sorted_idx = torch.sort(scores, descending=True)
            sorted_idx = sorted_idx.cpu().numpy()
            rel = np.array([1 if candidate_items[i] in pos_items else 0 for i in range(len(candidate_items))])
            rel_sorted = rel[sorted_idx]
            num_rel = int(rel_sorted.sum())
            if num_rel == 0:
                continue

            precisions = []
            hits = 0
            for rank, rel_flag in enumerate(rel_sorted, start=1):
                if rel_flag == 1:
                    hits += 1
                    precisions.append(hits / rank)
            map_list.append(np.mean(precisions) if precisions else 0.0)

            hit_ranks = np.where(rel_sorted == 1)[0]
            if len(hit_ranks) > 0:
                mrr_list.append(1.0 / (hit_ranks[0] + 1))

            for k in ks:
                topk_rel = rel_sorted[:k]
                if len(topk_rel) == 0:
                    continue
                prec = topk_rel.sum() / k
                rec = topk_rel.sum() / num_rel
                hit = 1.0 if topk_rel.sum() > 0 else 0.0

                discounts = 1.0 / np.log2(np.arange(2, 2 + len(topk_rel)))
                dcg = float((topk_rel * discounts).sum())

                ideal_rel = np.sort(rel_sorted)[::-1][:k]
                ideal_discounts = 1.0 / np.log2(np.arange(2, 2 + len(ideal_rel)))
                idcg = float((ideal_rel * ideal_discounts).sum())
                ndcg = dcg / idcg if idcg > 0 else 0.0

                precision_k[k].append(prec)
                recall_k[k].append(rec)
                ndcg_k[k].append(ndcg)
                hit_k[k].append(hit)

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"Precision@{k}"] = float(np.mean(precision_k[k])) if precision_k[k] else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(recall_k[k])) if recall_k[k] else 0.0
        metrics[f"NDCG@{k}"] = float(np.mean(ndcg_k[k])) if ndcg_k[k] else 0.0
        metrics[f"Hit@{k}"] = float(np.mean(hit_k[k])) if hit_k[k] else 0.0
    metrics["MAP"] = float(np.mean(map_list)) if map_list else 0.0
    metrics["MRR"] = float(np.mean(mrr_list)) if mrr_list else 0.0
    return metrics


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
def train_image_model(
    train_dl: DataLoader,
    val_dl: DataLoader,
    input_dim: int,
    proj_dim: int,
    num_users: int,
    user_id_map: Dict[int, int],
    dropout_rate: float,
    epoch_num: int,
    lr: float,
    log_name: str,
    topk_settings: Optional[TopKSettings],
    val_ds: Optional[MovieDataset],
    all_ds: Optional[MovieDataset],
    model_output_dir: pathlib.Path,
    use_scheduler: bool = True,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
) -> Tuple[ImageModel, pathlib.Path, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = _setup_logger(log_name)
    logger.info(
        "ImageModel | input_dim=%d | proj_dim=%d | num_users=%d | dropout=%.3f | epochs=%d | lr=%.6f",
        input_dim,
        proj_dim,
        num_users,
        dropout_rate,
        epoch_num,
        lr,
    )

    model = ImageModel(input_dim=input_dim, proj_dim=proj_dim, num_users=num_users, dropout_rate=dropout_rate)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        if use_scheduler
        else None
    )

    best_auc = 0.0
    best_metrics: Dict[str, float] = {}
    best_path = model_output_dir / "best_model_image_tower.pth"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epoch_num + 1):
        model.train()
        total_loss = 0.0
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm.tqdm(train_dl, desc=f"Training Epoch {epoch}")):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            optimizer.zero_grad()
            user_ids = batch["user_id"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device)

            user_id_indices = _map_user_ids_to_indices(user_ids, user_id_map, device)
            scores = model(user_id_indices, img_features)
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1
        avg_loss = total_loss / max(train_batches, 1)

        model.eval()
        y_true: List[float] = []
        y_pred: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(val_dl, desc=f"Validating Epoch {epoch}")):
                if max_val_batches and batch_idx >= max_val_batches:
                    break
                user_ids = batch["user_id"].to(device)
                img_features = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()

                user_id_indices = _map_user_ids_to_indices(user_ids, user_id_map, device)
                scores = model(user_id_indices, img_features)
                preds = torch.sigmoid(scores).cpu().numpy()

                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())

        auc = _safe_roc_auc(y_true, y_pred, logger)

        topk_metrics = {}
        if topk_settings and val_ds is not None and all_ds is not None:
            topk_metrics = compute_topk_metrics(
                model=model,
                val_ds=val_ds,
                all_ds=all_ds,
                topk_settings=topk_settings,
                logger=logger,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        metrics_str = " | ".join([f"{name}: {value:.4f}" for name, value in topk_metrics.items()]) if topk_metrics else ""
        base_log = (
            f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val AUC: {auc:.4f} | "
            f"Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}"
        )
        if metrics_str:
            base_log += f" | Top-K: {metrics_str}"
        logger.info(base_log)

        if scheduler is not None:
            prev_lr = current_lr
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                logger.info("Learning rate reduced from %.6f to %.6f", prev_lr, new_lr)

        if auc > best_auc:
            best_auc = auc
            best_metrics = {"Val AUC": best_auc}
            best_metrics.update(topk_metrics)
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved at %s", best_path)

    logger.info("Training finished. Best Val AUC=%.4f", best_auc)
    return model, best_path, best_metrics


# ---------------------------------------------------------------------------
# ARGUMENT PARSING / ENTRY POINT
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/validate the ImageModel tower")
    parser.add_argument("--train-csv", default=project_config.TRAIN_CSV)
    parser.add_argument("--val-csv", default=project_config.VAL_CSV)
    parser.add_argument("--all-csv", default=project_config.ALL_CSV)
    parser.add_argument("--text-features", default=project_config.TEXT_FEATURE_NPZ)
    parser.add_argument("--img-features", default=project_config.IMG_FEATURE_NPZ)
    parser.add_argument("--batch-size", type=int, default=project_config.TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--user-embedding-dim", type=int, default=64)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--sample-neg", type=int, default=100)
    parser.add_argument("--topk-max-users", type=int, default=0)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--log-name", default="image_model")
    parser.add_argument("--model-output-dir", default="model")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug: limit train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Debug: limit val batches per epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-topk", action="store_true", help="Skip expensive Top-K evaluation for faster debugging runs")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_ds = MovieDataset(args.train_csv, args.text_features, args.img_features)
    val_ds = MovieDataset(args.val_csv, args.text_features, args.img_features)
    all_ds = MovieDataset(args.all_csv, args.text_features, args.img_features)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    sample = next(iter(train_dl))
    img_dim = sample["img_features"].shape[1]
    num_users = len(all_ds.user_id_map)

    topk_settings: Optional[TopKSettings] = None
    if not args.skip_topk:
        topk_settings = TopKSettings(
            ks=args.topk,
            sample_neg=args.sample_neg,
            max_users=args.topk_max_users if args.topk_max_users > 0 else None,
        )

    model_output_dir = pathlib.Path(args.model_output_dir)
    model, best_path, best_metrics = train_image_model(
        train_dl=train_dl,
        val_dl=val_dl,
        input_dim=img_dim,
        proj_dim=args.proj_dim,
        num_users=num_users,
        user_id_map=all_ds.user_id_map,
        dropout_rate=args.dropout_rate,
        epoch_num=args.epochs,
        lr=args.lr,
        log_name=args.log_name,
        topk_settings=topk_settings,
        val_ds=val_ds,
        all_ds=all_ds,
        model_output_dir=model_output_dir,
        use_scheduler=not args.no_scheduler,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    if not args.skip_topk:
        print("\n=== Final Evaluation on Validation Set ===")
        final_metrics = compute_topk_metrics(
            model=model,
            val_ds=val_ds,
            all_ds=all_ds,
            topk_settings=topk_settings,
        )
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("Top-K evaluation skipped (--skip-topk).")
    print(f"Best checkpoint saved to: {best_path}")
    if best_metrics:
        print("Best metrics snapshot:")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
