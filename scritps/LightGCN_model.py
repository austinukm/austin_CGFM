import argparse
import logging
import pathlib
import random
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import tqdm
import scipy.sparse as sp

import project_config

# ------------------------------------------------------------------------------
# 1. Dataset & Graph Construction
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
        return None, None
    
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
        
        sample = {
            "movie_id": torch.tensor(movie_id, dtype=torch.long),
            "movie_idx": movie_idx_tensor,
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "user_idx": user_idx_tensor,
            "rating": rating_tensor,
            "label": label_tensor,
        }
        return sample

def build_graph(dataset, num_users, num_items, device):
    """
    Constructs the normalized adjacency matrix for LightGCN.
    Graph structure:
    | 0   R |
    | R^T 0 |
    """
    print("Building graph...")
    user_ids = []
    item_ids = []
    
    # Iterate over the dataframe to get positive interactions
    # We only use training data for the graph
    for row in dataset.data_df.itertuples():
        if row.label == 1:
            u_idx = dataset.user_id_map.get(row.user_id)
            i_idx = dataset.all_movie_idx_map.get(row.movie_id)
            if u_idx is not None and i_idx is not None:
                user_ids.append(u_idx)
                item_ids.append(i_idx)
                
    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    
    # Create sparse matrix R
    R = sp.coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=(num_users, num_items))
    
    # Build Adjacency Matrix A
    # A = [0, R]
    #     [R^T, 0]
    A = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    A = A.tolil()
    R = R.tolil()
    
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R.T
    A = A.todok()
    
    # Normalize: D^-1/2 A D^-1/2
    rowsum = np.array(A.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(A).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    # Convert to sparse tensor
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    graph = torch.sparse.FloatTensor(indices, values, shape).to(device)
    print("Graph built.")
    return graph

# ------------------------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------------------------

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, graph, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.graph = graph
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
    def forward_prop(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def forward(self, user_idx, item_idx):
        users_emb, items_emb = self.forward_prop()
        
        u_e = users_emb[user_idx]
        i_e = items_emb[item_idx]
        
        return (u_e * i_e).sum(dim=-1)

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

            candidate_indices = [all_ds.all_movie_idx_map.get(mid, 0) for mid in candidate_items]
            candidate_indices_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)

            user_idx = all_ds.user_id_map.get(user_id)
            if user_idx is None:
                continue

            user_tensor = torch.full((len(candidate_indices),), user_idx, dtype=torch.long, device=device)

            scores = model(user_tensor, candidate_indices_tensor)
                
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
    num_users,
    num_items,
    graph,
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
    logger.info(f"Arguments: {vars(args)}")
    
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        graph=graph,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers
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
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        
        for batch in tqdm.tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            user_idx = batch["user_idx"].to(device)
            movie_idx = batch["movie_idx"].to(device)
            labels = batch["label"].to(device)
            
            scores = model(user_idx, movie_idx)
            loss = criterion_bce(scores, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dl)
        
        # Validation (AUC)
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                user_idx = batch["user_idx"].to(device)
                movie_idx = batch["movie_idx"].to(device)
                labels = batch["label"].cpu().numpy()
                
                scores = model(user_idx, movie_idx)
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
        logger.info(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val AUC: {auc:.4f} | {metrics_str}")
        
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
    parser = argparse.ArgumentParser(description="Train LightGCN Model")
    parser.add_argument("--train-csv", default=project_config.TRAIN_CSV)
    parser.add_argument("--val-csv", default=project_config.VAL_CSV)
    parser.add_argument("--all-csv", default=project_config.ALL_CSV)
    parser.add_argument("--text-features", default=project_config.TEXT_FEATURE_NPZ)
    parser.add_argument("--img-features", default=project_config.IMG_FEATURE_NPZ)
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--sample-neg", type=int, default=100)
    parser.add_argument("--topk-max-users", type=int, default=0)
    
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--log-name", default="lightgcn_model")
    parser.add_argument("--device", default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Datasets
    train_ds = MovieDataset(args.train_csv, args.text_features, args.img_features)
    val_ds = MovieDataset(args.val_csv, args.text_features, args.img_features)
    all_ds = MovieDataset(args.all_csv, args.text_features, args.img_features)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    num_users = len(all_ds.user_id_map)
    num_items = len(all_ds.all_movie_idx_map)
    
    print(f"Data Loaded. Users: {num_users}, Items: {num_items}")
    
    # Build Graph
    graph = build_graph(train_ds, num_users, num_items, device)
    
    train_model(
        train_dl=train_dl,
        val_dl=val_dl,
        num_users=num_users,
        num_items=num_items,
        graph=graph,
        args=args,
        val_ds=val_ds,
        all_ds=all_ds
    )

if __name__ == "__main__":
    main()
