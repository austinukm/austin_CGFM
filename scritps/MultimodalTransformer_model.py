import argparse
import logging
import pathlib
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import tqdm

import project_config

# ------------------------------------------------------------------------------
# 1. Dataset (Reused from CGFM)
# ------------------------------------------------------------------------------

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, text_feature_path, img_feature_path):
        self.data_df = pd.read_csv(
            csv_path,
            sep=",",
            engine="python"
        )
        
        self.text_features = np.load(text_feature_path, allow_pickle=True)
        self.text_feature_dict = self.__construct_feature_dict(self.text_features)
        
        self.img_features = np.load(img_feature_path, allow_pickle=True)
        self.img_feature_dict = self.__construct_feature_dict(self.img_features)
        
        self.all_movie_idx_map = self.check_missing_features_get_allmovie_idx_map()
        self.user_id_map = self.get_user_id_dict()

        num_items = len(self.all_movie_idx_map)
        text_dim = self.text_features['features'].shape[1]
        img_dim = self.img_features['features'].shape[1]
        
        self.text_feature_tensor = torch.zeros((num_items, text_dim), dtype=torch.float)
        self.img_feature_tensor = torch.zeros((num_items, img_dim), dtype=torch.float)
        
        for mid, feat in self.text_feature_dict.items():
            idx = self.all_movie_idx_map[mid]
            self.text_feature_tensor[idx] = torch.tensor(feat, dtype=torch.float)
            
        for mid, feat in self.img_feature_dict.items():
            idx = self.all_movie_idx_map[mid]
            self.img_feature_tensor[idx] = torch.tensor(feat, dtype=torch.float)

    def construct_positive_dict(self):
        positive_dict = {}
        for row in self.data_df.itertuples():
            movie_id = row.movie_id
            label = row.label
            user_id = row.user_id
            if label == 1:
                if user_id not in positive_dict:
                    positive_dict[user_id] = set()
                positive_dict[user_id].add(movie_id)
        return positive_dict
    
    def __construct_feature_dict(self, feature_npz):
        feature_dict = {}
        ids = feature_npz['ids']
        features = feature_npz['features']
        for i, mid in enumerate(ids):
            if mid in feature_dict:
                pass 
            feature_dict[mid] = features[i]
        return feature_dict
        
    def __len__(self):
        return len(self.data_df)
        
    def check_missing_features_get_allmovie_idx_map(self):
        text_keys = list(self.text_feature_dict.keys())
        img_keys = list(self.img_feature_dict.keys())
        
        if set(text_keys) != set(img_keys):
             common_keys = set(text_keys) & set(img_keys)
             print(f"Warning: Feature mismatch. Using intersection of {len(common_keys)} items.")
             text_keys = list(common_keys)
        
        text_keys.sort()
        mid_idx_map = {mid: idx for idx, mid in enumerate(text_keys)}
        return mid_idx_map
    
    def get_user_id_dict(self):
        user_ids = self.data_df['user_id'].unique().tolist()
        user_ids.sort()
        user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        return user_id_map
    
    def get_features_for_movie_ids(self, movie_ids, device=None):
        idxs = []
        for mid in movie_ids:
            idx = self.all_movie_idx_map.get(mid)
            if idx is not None:
                idxs.append(idx)
        
        if not idxs:
            return None, None
            
        text_feats = self.text_feature_tensor[idxs]
        img_feats = self.img_feature_tensor[idxs]
        
        if device is not None:
            text_feats = text_feats.to(device, non_blocking=True)
            img_feats = img_feats.to(device, non_blocking=True)
            
        return text_feats, img_feats
    
    def get_interact_dict(self, sample_num=100):
        interact_dict = self.data_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        all_movie = set(self.all_movie_idx_map.keys())
        for user_id, interacted_movies in interact_dict.items():
            non_iteracted_movies = list(all_movie - interacted_movies)
            if len(non_iteracted_movies) > sample_num:
                neg_samples = random.sample(non_iteracted_movies, sample_num)
            else:
                neg_samples = non_iteracted_movies
                
            interact_dict[user_id] = {
                "interact_movies": interacted_movies,
                "non_interact_movies": neg_samples
            }
        return interact_dict
        
    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        movie_id = row["movie_id"]
        user_id = row["user_id"]
        
        movie_idx = self.all_movie_idx_map.get(movie_id, 0)
        user_idx = self.user_id_map.get(user_id, 0)
        
        movie_idx_tensor = torch.tensor(movie_idx, dtype=torch.long)
        user_idx_tensor = torch.tensor(user_idx, dtype=torch.long)
        
        rating_tensor = torch.tensor(row["rating"], dtype=torch.float)
        label_tensor = torch.tensor(row["label"], dtype=torch.float)
        
        text_feature = self.text_feature_tensor[movie_idx]
        img_feature = self.img_feature_tensor[movie_idx]
        
        sample = {
            "movie_id": torch.tensor(movie_id, dtype=torch.long),
            "movie_idx": movie_idx_tensor,
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "user_idx": user_idx_tensor,
            "rating": rating_tensor,
            "label": label_tensor,
            "text_features": text_feature,
            "img_features": img_feature,
        }
        return sample

# ------------------------------------------------------------------------------
# 2. Model (MultimodalTransformer Adapted)
# ------------------------------------------------------------------------------

class MultimodalTransformerRecommender(nn.Module):
    def __init__(self, text_input_dim, img_input_dim, num_users, num_items, hidden_dim=256, num_heads=4, num_layers=2, temperature=0.07, user_emb_dim=64, proj_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.img_proj = nn.Linear(img_input_dim, hidden_dim)
        
        # Positional encoding for 2 modalities (text, img)
        self.pos_encoder = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to common space for recommendation
        self.item_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, proj_dim)
        
        if user_emb_dim != proj_dim:
            self.user_proj = nn.Linear(user_emb_dim, proj_dim)
        else:
            self.user_proj = None
            
        self.temperature = temperature

    def forward_item(self, img_features, text_features, item_ids=None):
        text_emb = F.normalize(self.text_proj(text_features), p=2, dim=-1)
        img_emb = F.normalize(self.img_proj(img_features), p=2, dim=-1)
        
        # Stack: (batch_size, 2, hidden_dim)
        x = torch.stack((text_emb, img_emb), dim=1)
        x = x + self.pos_encoder
        x = self.transformer(x)
        
        # Mean pooling
        fused_hidden = x.mean(dim=1)
        
        fused = self.item_proj(fused_hidden)
        
        if item_ids is not None:
            fused = fused + self.item_emb(item_ids)
            
        return text_emb, img_emb, fused

    def score(self, user_ids, item_rep):
        u = self.user_emb(user_ids)
        if self.user_proj is not None:
            u = self.user_proj(u)
        scores = (u * item_rep).sum(dim=-1)
        return scores

    def forward(self, user_ids, item_ids, img_features, text_features):
        text_emb, img_emb, fused = self.forward_item(img_features, text_features, item_ids)
        scores = self.score(user_ids, fused)
        return scores, text_emb, img_emb, fused, None

    def contrastive_logits(self, text_rep, img_rep):
        t_n = F.normalize(text_rep, dim=-1)
        i_n = F.normalize(img_rep, dim=-1)
        logits = torch.matmul(t_n, i_n.t()) / self.temperature
        return logits

    @staticmethod
    def info_nce_loss(logits):
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        return (loss_t2i + loss_i2t) / 2

# ------------------------------------------------------------------------------
# 3. Training & Evaluation Utils
# ------------------------------------------------------------------------------

def _get_logger(model_name: str) -> logging.Logger:
    logs_dir = pathlib.Path("logs") / model_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_{timestamp}.log"
    
    logger = logging.getLogger(f"trainer.{model_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    logger.propagate = False
    logger.info(f"Logging to: {log_path}")
    return logger

def compute_topk_metrics(model, val_ds, all_ds, ks=[5, 10, 20], sample_neg_num=100, max_users=None, logger=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    positive_dict = val_ds.construct_positive_dict()
    neg_dict = all_ds.get_interact_dict(sample_num=sample_neg_num)

    ks = sorted(set(ks))

    precision_k = {k: [] for k in ks}
    recall_k = {k: [] for k in ks}
    ndcg_k = {k: [] for k in ks}
    hit_k = {k: [] for k in ks}
    map_list = []
    mrr_list = []

    user_ids = list(positive_dict.keys())
    if max_users and len(user_ids) > max_users:
        rng = random.Random(42)
        user_ids = rng.sample(user_ids, max_users)

    with torch.no_grad():
        for user_id in tqdm.tqdm(user_ids, desc="Computing Top-K metrics"):
            pos_items = positive_dict.get(user_id, set())
            if len(pos_items) == 0:
                continue
            neg_info = neg_dict.get(user_id)
            if not neg_info:
                continue

            candidate_items = list(pos_items) + neg_info["non_interact_movies"]
            if len(candidate_items) == 0:
                continue

            text_feats, img_feats = all_ds.get_features_for_movie_ids(candidate_items, device=device)
            if text_feats is None:
                continue
            
            candidate_indices = [all_ds.all_movie_idx_map.get(mid, 0) for mid in candidate_items]
            candidate_indices_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)

            user_idx = all_ds.user_id_map.get(user_id)
            if user_idx is None:
                continue

            user_tensor = torch.full((text_feats.size(0),), user_idx, dtype=torch.long, device=device)

            scores, _, _, _, _ = model(user_tensor, candidate_indices_tensor, img_feats, text_feats)
            if scores.dim() > 1:
                scores = scores.squeeze(-1)
                
            sorted_scores, sorted_idx = torch.sort(scores, descending=True)
            sorted_idx = sorted_idx.cpu().numpy()
            
            rel = np.array([1 if candidate_items[i] in pos_items else 0 for i in range(len(candidate_items))])
            rel_sorted = rel[sorted_idx]
            num_rel = rel_sorted.sum()
            if num_rel == 0:
                continue

            # MAP
            precisions = []
            hits = 0
            for rank, r in enumerate(rel_sorted, start=1):
                if r == 1:
                    hits += 1
                    precisions.append(hits / rank)
            map_list.append(np.mean(precisions) if len(precisions) > 0 else 0.0)

            # MRR
            hit_ranks = np.where(rel_sorted == 1)[0]
            if len(hit_ranks) > 0:
                mrr_list.append(1.0 / (hit_ranks[0] + 1))

            for k in ks:
                topk_rel = rel_sorted[:k]
                prec = topk_rel.sum() / k
                rec = topk_rel.sum() / num_rel
                hit = 1.0 if topk_rel.sum() > 0 else 0.0

                discounts = 1.0 / np.log2(np.arange(2, 2 + len(topk_rel)))
                dcg = (topk_rel * discounts).sum()

                ideal_rel = np.sort(rel_sorted)[::-1][:k]
                ideal_discounts = 1.0 / np.log2(np.arange(2, 2 + len(ideal_rel)))
                idcg = (ideal_rel * ideal_discounts).sum()
                ndcg = dcg / idcg if idcg > 0 else 0.0

                precision_k[k].append(prec)
                recall_k[k].append(rec)
                ndcg_k[k].append(ndcg)
                hit_k[k].append(hit)

    metrics = {}
    for k in ks:
        metrics[f"Precision@{k}"] = float(np.mean(precision_k[k])) if len(precision_k[k]) > 0 else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(recall_k[k])) if len(recall_k[k]) > 0 else 0.0
        metrics[f"NDCG@{k}"] = float(np.mean(ndcg_k[k])) if len(ndcg_k[k]) > 0 else 0.0
        metrics[f"Hit@{k}"] = float(np.mean(hit_k[k])) if len(hit_k[k]) > 0 else 0.0

    metrics["MAP"] = float(np.mean(map_list)) if len(map_list) > 0 else 0.0
    metrics["MRR"] = float(np.mean(mrr_list)) if len(mrr_list) > 0 else 0.0

    return metrics

def train_model(
    train_dl,
    val_dl,
    text_input_dim,
    img_input_dim,
    num_users,
    num_items,
    args,
    val_ds=None,
    all_ds=None,
):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = _get_logger(args.log_name)
    logger.info(f"Using device: {device}")
    
    model = MultimodalTransformerRecommender(
        text_input_dim=text_input_dim,
        img_input_dim=img_input_dim,
        num_users=num_users,
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        temperature=0.07,
        user_emb_dim=args.user_emb_dim,
        proj_dim=args.proj_dim
    )
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion_bce = nn.BCEWithLogitsLoss()
    
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    else:
        scheduler = None
        
    best_auc = 0.0
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)
    
    lambda_contrast = args.lambda_contrast
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_con_loss = 0.0
        
        for batch in tqdm.tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            user_idx = batch["user_idx"].to(device)
            movie_idx = batch["movie_idx"].to(device)
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device)
            
            scores, text_rep, img_rep, fused, g = model(user_idx, movie_idx, img_features, text_features)
            
            # Recommendation Loss
            rec_loss = criterion_bce(scores, labels)
            
            # Contrastive Loss (InfoNCE)
            contra_logits = model.contrastive_logits(text_rep, img_rep)
            con_loss = model.info_nce_loss(contra_logits)
            
            loss = rec_loss + lambda_contrast * con_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_con_loss += con_loss.item()
            
        avg_loss = total_loss / len(train_dl)
        avg_rec = total_rec_loss / len(train_dl)
        avg_con = total_con_loss / len(train_dl)
        
        # Validation (AUC)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                user_idx = batch["user_idx"].to(device)
                movie_idx = batch["movie_idx"].to(device)
                text_features = batch["text_features"].to(device)
                img_features = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()
                
                scores, _, _, _, _ = model(user_idx, movie_idx, img_features, text_features)
                preds = torch.sigmoid(scores).cpu().numpy()
                
                y_true.extend(labels)
                y_pred.extend(preds)
        
        auc = roc_auc_score(y_true, y_pred)
        
        # Top-K
        topk_metrics = {}
        if args.topk and val_ds and all_ds:
             topk_metrics = compute_topk_metrics(
                model=model,
                val_ds=val_ds,
                all_ds=all_ds,
                ks=args.topk,
                sample_neg_num=args.sample_neg,
                max_users=args.topk_max_users,
                logger=logger,
                device=device
            )
             
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in topk_metrics.items()])
        logger.info(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} (Rec: {avg_rec:.4f}, Con: {avg_con:.4f}) | Val AUC: {auc:.4f} | {metrics_str}")
        
        if scheduler:
            scheduler.step(auc)
            
        if auc > best_auc:
            best_auc = auc
            save_path = model_output_dir / f"best_model_{args.log_name}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
            
    return model

# ------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MultimodalTransformer Model")
    parser.add_argument("--train-csv", default=project_config.TRAIN_CSV)
    parser.add_argument("--val-csv", default=project_config.VAL_CSV)
    parser.add_argument("--all-csv", default=project_config.ALL_CSV)
    parser.add_argument("--text-features", default=project_config.TEXT_FEATURE_NPZ)
    parser.add_argument("--img-features", default=project_config.IMG_FEATURE_NPZ)
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--user-emb-dim", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    
    parser.add_argument("--lambda-contrast", type=float, default=0.1, help="Weight for contrastive loss")
    
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--sample-neg", type=int, default=100)
    parser.add_argument("--topk-max-users", type=int, default=0)
    
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--log-name", default="multimodal_transformer_model")
    parser.add_argument("--device", default=None, help="Device to use (e.g. 'cpu', 'cuda')")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Datasets
    train_ds = MovieDataset(args.train_csv, args.text_features, args.img_features)
    val_ds = MovieDataset(args.val_csv, args.text_features, args.img_features)
    all_ds = MovieDataset(args.all_csv, args.text_features, args.img_features)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    sample = next(iter(train_dl))
    text_dim = sample["text_features"].shape[1]
    img_dim = sample["img_features"].shape[1]
    num_users = len(all_ds.user_id_map)
    num_items = len(all_ds.all_movie_idx_map)
    
    print(f"Data Loaded. Users: {num_users}, Items: {num_items}, Text Dim: {text_dim}, Img Dim: {img_dim}")
    
    train_model(
        train_dl=train_dl,
        val_dl=val_dl,
        text_input_dim=text_dim,
        img_input_dim=img_dim,
        num_users=num_users,
        num_items=num_items,
        args=args,
        val_ds=val_ds,
        all_ds=all_ds
    )

if __name__ == "__main__":
    main()
