import argparse
import logging
import pathlib
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import tqdm

import project_config
class MoiveDataset(torch.utils.data.Dataset):
    def __init__(self,csv_path,text_feature_path,img_feature_path):
        self.data_df = pd.read_csv(
            csv_path,
            sep=",",
            engine="python"
        )
        
        self.text_features = np.load(text_feature_path, allow_pickle=True)
        
        ###movie_id :: feature
        self.text_feature_dict = self.__construct_feature_dict(self.text_features)
        self.img_features = np.load(img_feature_path, allow_pickle=True)
        self.img_feature_dict = self.__construct_feature_dict(self.img_features)
        self.all_movie_idx_map = self.check_missing_features_get_allmovie_idx_map()
        self.user_id_map = self.get_user_id_dict()

        # 缓存为 tensor，便于批量索引
        self.text_feature_tensor = torch.from_numpy(self.text_features["features"]).float()
        
    
    def construct_positive_dict(self):
        """构建用户的正样本字典 {user_id: set(movie_ids)}"""
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
    
    
    def __construct_feature_dict(self,feature_npz):
        feature_dict = {}
        ids = feature_npz['ids']
        features = feature_npz['features']
        for i ,mid in enumerate(ids):
            #####check_repea###
            if mid in feature_dict:
                raise ValueError(f"AustinERROR duplicate movie id found:{mid}")
            feature_dict[mid] = features[i]
        return feature_dict
        
    def __len__(self):
        return len(self.data_df)
        
    def check_missing_features_get_allmovie_idx_map(self):
        text_keys = list(self.text_feature_dict.keys())
        img_keys = list(self.img_feature_dict.keys())
        if text_keys != img_keys:
            missing_in_text = set(img_keys) - set(text_keys)    
            missing_in_img = set(text_keys) - set(img_keys)
            raise ValueError(f"AustinERROR Missing features found. Missing in text: {missing_in_text}, Missing in img: {missing_in_img}")
        else:
            mid_idx_map = {}
            for idx,mid in enumerate(text_keys):
                mid_idx_map[mid] = idx
            return mid_idx_map
    
    def get_user_id_dict(self):
        user_ids = self.data_df['user_id'].unique().tolist()
        user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        return user_id_map
    
    
    def get_item_by_movie_id(self,movie_id):
        idx = self.all_movie_idx_map[movie_id]
        img_feature = torch.tensor(self.img_features['features'][idx],dtype=torch.float)
        text_feature = torch.tensor(self.text_features['features'][idx],dtype=torch.float)  
        resutl_dict = {
            "map_idx" : idx,
            "img_features" : img_feature,
            "text_features" : text_feature,
        }  
        return resutl_dict

    def get_text_tensor_for_movie_ids(self, movie_ids, device=None):
        idxs = []
        for mid in movie_ids:
            idx = self.all_movie_idx_map.get(mid)
            if idx is not None:
                idxs.append(idx)
        if not idxs:
            return None
        feats = self.text_feature_tensor[idxs]
        if device is not None:
            feats = feats.to(device, non_blocking=True)
        return feats
    
    def get_interact_dict(self,sample_num = 100):
        interact_dict = self.data_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        all_movie = set(self.all_movie_idx_map.keys())
        for user_id , interacted_movies in interact_dict.items():
            non_iteracted_movies = all_movie - interacted_movies
            interact_dict[user_id] = {
                "interact_movies" : interacted_movies,
                "non_interact_movies" : random.sample(list(non_iteracted_movies), min(sample_num,len(non_iteracted_movies)))
            }
        return interact_dict
        
        
    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        movie_id = row["movie_id"]
        movie_id_tensor = torch.tensor(movie_id,dtype=torch.long)
        user_id = row["user_id"]
        user_id_tensor = torch.tensor(user_id,dtype=torch.long)
        rating_tensor = torch.tensor(row["rating"],dtype=torch.float)
        label_tensor = torch.tensor(row["label"],dtype=torch.float)
        text_feature  = torch.tensor(self.text_feature_dict[movie_id],dtype=torch.float)
        img_feature  = torch.tensor(self.img_feature_dict[movie_id],dtype=torch.float)
        sample = {
            "movie_id" : movie_id_tensor,
            "user_id" : user_id_tensor,
            "rating" : rating_tensor,
            "label" : label_tensor,
            "text_features" : text_feature,
            "img_features" : img_feature,
        }
        return sample
    
class TextModel(nn.Module):
    
    def __init__(self,input_dim,proj_dim,num_users,dropout_rate,user_embeding_dim = 64):
        super().__init__()
        self.user_embeding = nn.Embedding(num_users , user_embeding_dim)
        self.user_proj = nn.Linear(user_embeding_dim,proj_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512,proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        
    def forward(self,user_id,text_features):
        u = self.user_embeding(user_id)
        u = self.user_proj(u)
        t = self.fc(text_features)
        score = torch.sum(u*t,dim=1)            
        return score



class ImageModle(nn.Module):
    
    def __init__(self,input_dim,proj_dim,num_users,dropout_rate,user_embeding_dim = 64): 
        super().__init__()
        self.user_embeding = nn.Embedding(num_users , user_embeding_dim)
        self.user_proj = nn.Linear(user_embeding_dim,proj_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512,proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        
    def forward(self,user_id,img_features):
        u = self.user_embeding(user_id)
        u = self.user_proj(u)
        t = self.fc(img_features)
        score = torch.sum(u*t,dim=1)            
        return score








def _map_user_ids_to_indices(user_ids, user_id_map, device):
    indices = []
    for uid in user_ids:
        uid_int = int(uid.item())
        if uid_int not in user_id_map:
            raise ValueError(f"AustinERROR Unknown user_id {uid_int} encountered during training.")
        indices.append(user_id_map[uid_int])
    return torch.tensor(indices, dtype=torch.long, device=device)


def _get_logger(model_name: str) -> logging.Logger:
    """创建日志记录器"""
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


def train_model(
    train_dl,
    val_dl,
    input_dim,
    proj_dim,
    num_users,
    user_id_map,
    dropout_rate,
    epoch_num,
    lr,
    use_scheduler=True,
    topk_k_list=None,
    topk_max_users=None,
    val_ds=None,
    all_ds=None,
    log_name="text_model",
    sample_neg=100,
):
    """训练TextModel模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = _get_logger(log_name or "text_model")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: TextModel | input_dim: {input_dim} | proj_dim: {proj_dim} | num_users: {num_users} | dropout_rate: {dropout_rate} | epochs: {epoch_num} | lr: {lr}")
    
    model = TextModel(input_dim=input_dim, proj_dim=proj_dim, num_users=num_users, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_logitsbce = nn.BCEWithLogitsLoss()
    model.to(device)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    else:
        scheduler = None
    
    best_auc = 0.0
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, epoch_num + 1):
        # --------------------
        # 🔹 Train
        # --------------------
        model.train()
        total_loss = 0.0
        
        for batch in tqdm.tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            user_ids = batch["user_id"].to(device)  # batch of user_ids
            text_features = batch["text_features"].to(device)
            labels = batch["label"].to(device)

            # 将user_id转换为索引
            user_id_indices = _map_user_ids_to_indices(user_ids, user_id_map, device)

            scores = model(user_id_indices, text_features)
            loss = criterion_logitsbce(scores, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # --------------------
        # 🔹 Validation
        # --------------------
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                user_ids = batch["user_id"].to(device)
                text_features = batch["text_features"].to(device)
                labels = batch["label"].cpu().numpy()

                user_id_indices = _map_user_ids_to_indices(user_ids, user_id_map, device)
                scores = model(user_id_indices, text_features)
                preds = torch.sigmoid(scores).cpu().numpy()

                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())

        auc = roc_auc_score(y_true, y_pred)

        # --------------------
        # 🔹 Top-K 推荐评估（user-wise）
        # --------------------
        topk_metrics = {}
        if topk_k_list and val_ds is not None and all_ds is not None:
            topk_metrics = compute_topk_metrics(
                model=model,
                val_ds=val_ds,
                all_ds=all_ds,
                ks=topk_k_list,
                sample_neg_num=sample_neg,
                max_users=topk_max_users,
                logger=logger
            )

        current_lr = optimizer.param_groups[0]["lr"]
        metrics_str = (
            " | ".join([f"{name}: {value:.4f}" for name, value in topk_metrics.items()])
            if topk_metrics
            else ""
        )
        base_log = (
            f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | "
            f"Val AUC: {auc:.4f} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}"
        )
        if metrics_str:
            base_log += f" | Top-K: {metrics_str}"
        logger.info(base_log)

        if scheduler is not None:
            previous_lr = current_lr
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < previous_lr:
                logger.info(f"📉 Learning rate reduced from {previous_lr:.6f} to {new_lr:.6f}")

        if auc > best_auc:
            best_auc = auc
            best_path = model_output_dir / "best_model_text_model.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"✅ New best model saved at {best_path} with AUC={best_auc:.4f}")

    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model
    




def compute_topk_metrics(model, val_ds, all_ds, ks=[5, 10, 20], sample_neg_num=100, max_users=None, logger=None):
    """
    计算多种 Top-K 指标（Precision@K, Recall@K, NDCG@K, MAP, MRR）
    基于每个用户完整候选集（正样本 + 负样本），不依赖 DataLoader 的 batch 顺序。
    """
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
                if logger:
                    logger.debug(f"Skip user {user_id}: no positive items.")
                continue
            neg_info = neg_dict.get(user_id)
            if not neg_info:
                if logger:
                    logger.debug(f"Skip user {user_id}: no negative samples available.")
                continue

            candidate_items = list(pos_items) + neg_info["non_interact_movies"]
            if len(candidate_items) == 0:
                if logger:
                    logger.debug(f"Skip user {user_id}: no candidate movies.")
                continue

            text_feature_tensor = all_ds.get_text_tensor_for_movie_ids(candidate_items, device=device)
            if text_feature_tensor is None or text_feature_tensor.size(0) == 0:
                if logger:
                    logger.debug(f"Skip user {user_id}: unable to fetch text features.")
                continue

            user_idx = all_ds.user_id_map.get(user_id)
            if user_idx is None:
                if logger:
                    logger.debug(f"Skip user {user_id}: user id not found in map.")
                continue

            user_tensor = torch.full((text_feature_tensor.size(0),), user_idx, dtype=torch.long, device=device)

            scores = model(user_tensor, text_feature_tensor).squeeze(-1)
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


def get_recomend_score(model, val_ds, all_ds, k=10, sample_num=100):
    """
    兼容旧接口，返回单一 K 的 Recall / Precision
    """
    metrics = compute_topk_metrics(
        model=model,
        val_ds=val_ds,
        all_ds=all_ds,
        ks=[k],
        sample_neg_num=sample_num
    )
    avg_recall = metrics.get(f"Recall@{k}", 0.0)
    avg_precision = metrics.get(f"Precision@{k}", 0.0)
    return avg_recall, avg_precision, {}
            

def parse_args():
    parser = argparse.ArgumentParser(description="Debug TextModel training & evaluation")
    parser.add_argument("--train-csv", default=project_config.TRAIN_CSV, help="Path to training CSV file")
    parser.add_argument("--val-csv", default=project_config.VAL_CSV, help="Path to validation CSV file")
    parser.add_argument("--all-csv", default=project_config.ALL_CSV, help="Path to full ratings CSV (for negative sampling)")
    parser.add_argument("--text-features", default=project_config.TEXT_FEATURE_NPZ, help="Path to text feature npz")
    parser.add_argument("--img-features", default=project_config.IMG_FEATURE_NPZ, help="Path to image feature npz")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--proj-dim", type=int, default=256, help="Projection dimension for text model")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate in text tower")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20], help="List of K values for Top-K metrics")
    parser.add_argument("--sample-neg", type=int, default=100, help="Number of negatives to sample per user for Top-K evaluation")
    parser.add_argument("--topk-max-users", type=int, default=0, help="Max number of users per epoch for Top-K metrics (0=all)")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable ReduceLROnPlateau scheduler")
    parser.add_argument("--log-name", default="text_model", help="Logger/model identifier")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build datasets and dataloaders
    train_ds = MoiveDataset(args.train_csv, args.text_features, args.img_features)
    val_ds = MoiveDataset(args.val_csv, args.text_features, args.img_features)
    all_ds = MoiveDataset(args.all_csv, args.text_features, args.img_features)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample = next(iter(train_dl))
    text_dim = sample["text_features"].shape[1]
    num_users = len(all_ds.user_id_map)

    model = train_model(
        train_dl=train_dl,
        val_dl=val_dl,
        input_dim=text_dim,
        proj_dim=args.proj_dim,
        num_users=num_users,
        user_id_map=all_ds.user_id_map,
        dropout_rate=args.dropout_rate,
        epoch_num=args.epochs,
        lr=args.lr,
        use_scheduler=not args.no_scheduler,
        topk_k_list=args.topk,
        topk_max_users=args.topk_max_users if args.topk_max_users > 0 else None,
        val_ds=val_ds,
        all_ds=all_ds,
        log_name=args.log_name,
        sample_neg=args.sample_neg,
    )

    print("\n=== Final Top-K Evaluation ===")
    final_metrics = compute_topk_metrics(
        model=model,
        val_ds=val_ds,
        all_ds=all_ds,
        ks=args.topk,
        sample_neg_num=args.sample_neg,
        max_users=args.topk_max_users if args.topk_max_users > 0 else None,
    )
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()