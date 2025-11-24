import torch
from torch.utils.data import Dataset,DataLoader
import argparse
import logging
from datetime import datetime
import pandas as pd
import pathlib
import numpy as np
from tqdm import tqdm
import project_config
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from collections import defaultdict
class MovieDataset(torch.utils.data.Dataset):
    def __init__(self,csv_path,text_features,img_features):
       self.data = pd.read_csv(
           csv_path,
           sep=",",
           engine = "python"
       )
       self.text_features = np.load(text_features, allow_pickle=True)
       self.text_feature_dict = self.__construct_feature_dict(self.text_features)  
       self.img_features = np.load(img_features, allow_pickle=True)
       self.img_feature_dict = self.__construct_feature_dict(self.img_features)
    
    def __construct_feature_dict(self,features_npz):
        feature_dict = {}
        ids = features_npz['ids']
        features = features_npz['features']
        for i,mid in enumerate(ids):
            ####check_repeat 
            if mid in feature_dict:
                raise ValueError(f"AustinERROR Duplicate movie id found: {mid}")
            feature_dict[mid] = features[i]
        return feature_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]       
        movie_id = row["movie_id"]
        movie_id_tensor = torch.tensor(int(movie_id),dtype=torch.long)
        user_id = row["user_id"]
        user_id_tensor = torch.tensor(int(user_id),dtype=torch.long)
        rating_tensor = torch.tensor(int(row["rating"]),dtype=torch.float)
        text_features = torch.tensor(self.text_feature_dict[movie_id],dtype=torch.float)
        img_features = torch.tensor(self.img_feature_dict[movie_id],dtype=torch.float)
        label = torch.tensor(row["label"],dtype=torch.float)
        result_dict = {
            "user_id": user_id_tensor,
            "movie_id": movie_id_tensor,
            "rating": rating_tensor,
            "label": label,
            "text_features": text_features,
            "img_features": img_features
        }
        return result_dict  
    def get_movie_text_feature(self,movie_id):
        if movie_id in self.text_feature_dict:
            return self.text_feature_dict[movie_id]
        else:
            raise ValueError(f"Movie ID {movie_id} not found in text features.")
    def get_movie_img_feature(self,movie_id):
        if movie_id in self.img_feature_dict:
            return self.img_feature_dict[movie_id]
        else:
            raise ValueError(f"Movie ID {movie_id} not found in image features.")





class MultimodalRecModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.fc(x)

class MultimodalAddProjectionGateModel(nn.Module):
    def __init__(self, text_input_dim, img_input_dim,projection_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, projection_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, projection_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.gate = nn.Sequential(
            nn.Linear(projection_dim*4, projection_dim*2),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim*2,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,128),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,text_features,img_features,return_emb =False):
        text_emb = F.normalize(self.text_proj(text_features),p=2, dim=-1)
        img_emb = F.normalize(self.img_proj(img_features),p=2, dim=-1)
        gate = self.gate(torch.cat((text_emb, img_emb), dim=-1)) 
        fused= (gate * text_emb) + ((1 - gate) * img_emb)
        output = self.classifier(fused)
        if return_emb:
            return output, text_emb, img_emb
        return output


class MultimodalAddProjectionModel(nn.Module):
    """原始的投影模型（不带Gate，带对比损失）"""
    def __init__(self, text_input_dim, img_input_dim, projection_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
            
     
    def forward(self, text_features, img_features, return_emb=False):
        text_emb = self.text_proj(text_features)
        img_emb = self.img_proj(img_features)
        combined = torch.cat((text_emb, img_emb), dim=-1)
        output = self.classifier(combined)
        if return_emb:
            return text_emb, img_emb, output
        else:
            return output


class average_fusion_model(nn.Module):
    ####  feature = img + text-> 512 /2 
    def __init__(self, text_input_dim, img_input_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_input_dim,img_input_dim)
        self.aim_dim = img_input_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.aim_dim,self.aim_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.aim_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,text_features,img_features,return_emb=False):
        self.text_features = self.text_proj(text_features)
        average_features = (self.text_features + img_features) / 2
        output = self.classifier(average_features)
        if return_emb:
            return output, self.text_features, img_features
        return output


def train_average_fusion_model(train_dl,val_dl,text_dim,img_dim,epoch_num,lr=5e-4,use_scheduler=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger("average_fusion")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: average_fusion_model | text_dim: {text_dim} | img_dim: {img_dim} | epochs: {epoch_num} | lr: {lr}")
    model = average_fusion_model(text_input_dim=text_dim, img_input_dim=img_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_bce = nn.BCELoss()
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    else:
        scheduler = None
    best_auc = 0.0
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)
    for epoch in range(1, epoch_num + 1):
        model.train()
        total_loss = 0.0
        total_bce_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            output, text_features, img_features = model(text_features, img_features, return_emb=True)
            loss_bce = criterion_bce(output, labels)
            loss_bce.backward()
            optimizer.step()
            total_loss += loss_bce.item()
            total_bce_loss += loss_bce.item()
        avg_train_loss = total_loss / len(train_dl)
        avg_bce_loss = total_bce_loss / len(train_dl)
        #validation
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                text_features = batch["text_features"].to(device)
                img_features = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()
                outputs = model(text_features, img_features, return_emb=False)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())
        auc = roc_auc_score(y_true, y_pred)
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
        current_lr = optimizer.param_groups[0]['lr']
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} (BCE: {avg_bce_loss:.4f}) | Val {metrics_str} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}")
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_output_dir / "best_model_average_fusion.pth")
            logger.info(f"✅ New best model saved (average_fusion) with AUC={best_auc:.4f}")
    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model


class HeavyTransformerModel(nn.Module):
    def __init__(self,text_input_dim,img_input_dim,hidden_dim,num_heads,num_layers):
        super().__init__()
        self.text_proj = nn.Linear(text_input_dim,hidden_dim)
        self.img_proj = nn.Linear(img_input_dim,hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1,2,hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_dim,
            nhead = num_heads,
            dim_feedforward = hidden_dim * 2,
            dropout = 0.3,
            batch_first = True,
            activation = "relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,text_features,img_features,return_emb=False):
        text_emb = F.normalize(self.text_proj(text_features),p=2,dim=-1)
        img_emb = F.normalize(self.img_proj(img_features),p=2,dim=-1)
        x = torch.stack((text_emb,img_emb),dim=1)
        x = x + self.pos_encoder
        x = self.transformer(x)
        fused = x.mean(dim=1)
        out = self.classifier(fused)
        if return_emb:
            return out,text_emb,img_emb
        return out
    
def train_HeavyTransformerModel(train_dl,val_dl,text_dim,img_dim,hidden_dim,num_heads,num_layers,epoch_num,lr=5e-4,use_scheduler=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger("heavy_transformer")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: HeavyTransformerModel | text_dim: {text_dim} | img_dim: {img_dim} | hidden_dim: {hidden_dim} | num_heads: {num_heads} | num_layers: {num_layers} | epochs: {epoch_num} | lr: {lr}")
    model = HeavyTransformerModel(text_input_dim=text_dim, img_input_dim=img_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_bce = nn.BCELoss()
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    else:
        scheduler = None
    best_auc = 0.0
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)
    for epoch in range(1, epoch_num + 1):
        model.train()
        total_loss = 0.0
        total_bce_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            output, text_emb, img_emb = model(text_features, img_features, return_emb=True)
            loss_bce = criterion_bce(output, labels)
            loss_bce.backward()
            optimizer.step()
            total_loss += loss_bce.item()
            total_bce_loss += loss_bce.item()
        avg_train_loss = total_loss / len(train_dl)
        avg_bce_loss = total_bce_loss / len(train_dl)
        #validation
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                text_features = batch["text_features"].to(device)
                img_features = batch["img_features"].to(device)
                labels =  batch["label"].cpu().numpy()
                outputs = model(text_features, img_features, return_emb=False)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())

        auc = roc_auc_score(y_true, y_pred)
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
        current_lr = optimizer.param_groups[0]['lr']
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} (BCE: {avg_bce_loss:.4f}) | Val {metrics_str} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}")
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_output_dir / "best_model_heavy_transformer.pth")
            logger.info(f"✅ New best model saved (heavy_transformer) with AUC={best_auc:.4f}")
    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model

class MultimodalTransformerModel(nn.Module):
    """多模态Transformer模型"""
    def __init__(self, text_input_dim, img_input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.img_proj = nn.Linear(img_input_dim, hidden_dim)
        
        # 位置编码（2个模态：text和img）
        self.pos_encoder = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


        
    def forward(self, text_features, img_features, return_emb=False):
        text_emb = F.normalize(self.text_proj(text_features), p=2, dim=-1)
        img_emb = F.normalize(self.img_proj(img_features), p=2, dim=-1)
        # 堆叠成序列: (batch_size, 2, hidden_dim)
        x = torch.stack((text_emb, img_emb), dim=1)
        # 添加位置编码
        x = x + self.pos_encoder
        # Transformer编码
        x = self.transformer(x)
        # 平均池化得到融合特征
        fused = x.mean(dim=1)
        out = self.classifier(fused)
        if return_emb:
            return out, text_emb, img_emb
        return out


def train_MultimodalTransformerModel(train_dl, val_dl, text_dim, img_dim, hidden_dim, num_heads, num_layers, lambda_align, epoch_num, lr=5e-4, use_scheduler=True):
    """训练多模态Transformer模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger("transformer")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: MultimodalTransformerModel | text_dim: {text_dim} | img_dim: {img_dim} | hidden_dim: {hidden_dim} | num_heads: {num_heads} | num_layers: {num_layers} | lambda_align: {lambda_align} | epochs: {epoch_num} | lr: {lr}")
    
    model = MultimodalTransformerModel(text_input_dim=text_dim, img_input_dim=img_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_bce = nn.BCELoss()
    
    # 添加学习率调度器
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
        total_bce_loss = 0.0
        total_align_loss = 0.0
        align_count = 0
        
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            output, text_emb, img_emb = model(text_features, img_features, return_emb=True)
            loss_bce = criterion_bce(output, labels)
            # 使用余弦相似度的对齐损失
            align_loss = 1 - F.cosine_similarity(text_emb, img_emb, dim=-1).mean()
            loss = loss_bce + lambda_align * align_loss
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total_bce_loss += loss_bce.item()
            total_align_loss += align_loss.item()
            align_count += 1
        
        avg_train_loss = total_loss / len(train_dl)
        avg_bce_loss = total_bce_loss / len(train_dl)
        avg_align_loss = total_align_loss / align_count if align_count > 0 else 0.0
        
        # --------------------
        # 🔹 Validation
        # --------------------
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                text = batch["text_features"].to(device)
                img = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()
                outputs = model(text, img, return_emb=False)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())
        
        # 计算 AUC
        auc = roc_auc_score(y_true, y_pred)
        
        # 计算推荐系统指标
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
        
        # --------------------
        # 🔹 打印与保存
        # --------------------
        current_lr = optimizer.param_groups[0]['lr']
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} (BCE: {avg_bce_loss:.4f}, Align: {avg_align_loss:.4f}) | Val {metrics_str} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}")
        
        # 学习率调度
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_output_dir / "best_model_transformer.pth")
            logger.info(f"✅ New best model saved (transformer) with AUC={best_auc:.4f}")
    
    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model




def train_MultimodalProjectionGateModel(train_dl,val_dl,text_dim,img_dim,proj_dim,lambda_align,epoch_num,lr=5e-4,use_scheduler=True):
    """训练带Gate机制和对比损失的投影模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger("projectiongate")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: MultimodalAddProjectionGateModel | text_dim: {text_dim} | img_dim: {img_dim} | proj_dim: {proj_dim} | lambda_align: {lambda_align} | epochs: {epoch_num} | lr: {lr}")
    
    model = MultimodalAddProjectionGateModel(text_input_dim=text_dim, img_input_dim=img_dim, projection_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_bce = nn.BCELoss()
    
    # 添加学习率调度器
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
        total_bce_loss = 0.0
        total_align_loss = 0.0
        align_count = 0
        
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            output, text_emb, img_emb = model(text_features, img_features, return_emb=True)
            loss_bce = criterion_bce(output, labels)
            # 使用余弦相似度的对齐损失
            align_loss = 1 - F.cosine_similarity(text_emb, img_emb, dim=-1).mean()
            loss = loss_bce + lambda_align * align_loss
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total_bce_loss += loss_bce.item()
            total_align_loss += align_loss.item()
            align_count += 1
        
        avg_train_loss = total_loss / len(train_dl)
        avg_bce_loss = total_bce_loss / len(train_dl)
        avg_align_loss = total_align_loss / align_count if align_count > 0 else 0.0
        
        # --------------------
        # 🔹 Validation
        # --------------------
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                text = batch["text_features"].to(device)
                img = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()
                outputs = model(text, img, return_emb=False)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())
        
        # 计算 AUC
        auc = roc_auc_score(y_true, y_pred)
        
        # 计算推荐系统指标
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
        
        # --------------------
        # 🔹 打印与保存
        # --------------------
        current_lr = optimizer.param_groups[0]['lr']
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} (BCE: {avg_bce_loss:.4f}, Align: {avg_align_loss:.4f}) | Val {metrics_str} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}")
        
        # 学习率调度
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_output_dir / "best_model_projectiongate.pth")
            logger.info(f"✅ New best model saved (projectiongate) with AUC={best_auc:.4f}")
    
    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model

class MultimodalProjectionModel(nn.Module):
    def __init__(self, text_input_dim, img_input_dim,projection_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim,512),
            nn.ReLU(),
            nn.Linear(512,projection_dim),
            nn.LayerNorm(projection_dim)
            
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim,512),
            nn.ReLU(),
            nn.Linear(512,projection_dim),
            nn.LayerNorm(projection_dim)

        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim*2,512),
            nn.ReLU(),
            nn.Linear(512,projection_dim),
            nn.LayerNorm(projection_dim),
        )

        
        

    def forward(self,text_features,img_features,return_emb =False):
        text_emb = self.text_proj(text_features)
        img_emb = self.img_proj(img_features)
        combined = torch.cat((text_emb, img_emb), dim=-1)
        output = self.classifier(combined)
        if return_emb:
            return text_emb, img_emb, output
        else:
            return output



def constractive_loss(text_emb, img_emb, temperature=0.2):
    text_emb = F.normalize(text_emb, dim=-1)
    img_emb = F.normalize(img_emb, dim=-1)
    logits = text_emb @ img_emb.T / temperature
    labels = torch.arange(text_emb.size(0), device=text_emb.device)
    loss_text_to_img = F.cross_entropy(logits, labels)
    loss_img_to_text = F.cross_entropy(logits.T, labels)
    loss = (loss_text_to_img + loss_img_to_text) / 2
    return loss



train_ds = (MovieDataset(
    csv_path=project_config.TRAIN_CSV,
    text_features=project_config.TEXT_FEATURE_NPZ,
    img_features=project_config.IMG_FEATURE_NPZ
))  
val_ds = (MovieDataset(
    csv_path=project_config.VAL_CSV,
    text_features=project_config.TEXT_FEATURE_NPZ,
    img_features=project_config.IMG_FEATURE_NPZ
))
test_ds = (MovieDataset(
    csv_path=project_config.TEST_CSV,
    text_features=project_config.TEXT_FEATURE_NPZ,
    img_features=project_config.IMG_FEATURE_NPZ
))
all_ds = (MovieDataset(
    csv_path=project_config.ALL_CSV,
    text_features=project_config.TEXT_FEATURE_NPZ,
    img_features=project_config.IMG_FEATURE_NPZ
))

train_df = train_ds.ratings
all_df = all_ds.ratings
train_dl = DataLoader(train_ds,batch_size=32,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=32,shuffle=False)


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_input = next(iter(train_dl))
    print(sample_input)
    print(sample_input["text_features"].shape)
    print(sample_input["img_features"].shape)
    input_dim=sample_input["text_features"].shape[1]+sample_input["img_features"].shape[1]
    model = MultimodalRecModel(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_num = 10
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dl):
            optimizer.zero_grad()
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            outputs = model(torch.cat((text_features, img_features), dim=-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dl)}")
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_output_dir / f"multimodal_rec_model{epoch_num}.pth")



def train_MultimodalAddProjectionModel(train_dl,val_dl,text_dim,img_dim,proj_dim,lambda_align,epoch_num,lr=5e-4,use_scheduler=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger("projection")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: MultimodalAddProjectionModel | text_dim: {text_dim} | img_dim: {img_dim} | proj_dim: {proj_dim} | lambda_align: {lambda_align} | epochs: {epoch_num} | lr: {lr}")
    
    model = MultimodalAddProjectionModel(text_input_dim=text_dim,img_input_dim=img_dim,projection_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce_loss = nn.BCELoss()
    
    # 添加学习率调度器
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
        total_bce_loss = 0.0
        total_align_loss = 0.0
        align_count = 0
        
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            text = batch["text_features"].to(device)
            img = batch["img_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            pred, text_emb, img_emb = model(text, img, return_emb=True)
            loss_bce = bce_loss(pred, labels)
            pos_mask = labels.squeeze() > 0.5
            if pos_mask.sum() > 1:
                loss_align = constractive_loss(text_emb[pos_mask], img_emb[pos_mask])
                total_align_loss += loss_align.item()
                align_count += 1
            else:
                loss_align = torch.tensor(0.0, device=device)
            
            loss = loss_bce + lambda_align * loss_align
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total_bce_loss += loss_bce.item()
        
        avg_train_loss = total_loss / len(train_dl)
        avg_bce_loss = total_bce_loss / len(train_dl)
        avg_align_loss = total_align_loss / align_count if align_count > 0 else 0.0
        
        # --------------------
        # 🔹 Validation
        # --------------------
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                text = batch["text_features"].to(device)
                img = batch["img_features"].to(device)
                labels = batch["label"].cpu().numpy()
                outputs = model(text, img, return_emb=False)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())
        
        # 计算 AUC
        auc = roc_auc_score(y_true, y_pred)
        
        # 计算推荐系统指标
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
        
        # --------------------
        # 🔹 打印与保存
        # --------------------
        current_lr = optimizer.param_groups[0]['lr']
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} (BCE: {avg_bce_loss:.4f}, Align: {avg_align_loss:.4f}) | Val {metrics_str} | Best AUC: {best_auc:.4f} | LR: {current_lr:.6f}")
        
        # 学习率调度
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(auc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_output_dir / "best_model_projection.pth")
            logger.info(f"✅ New best model saved (projection) with AUC={best_auc:.4f}")
    
    logger.info(f"Training finished. Best AUC={best_auc:.4f}")
    return model


def test_MultimodalAddProjectionModel(model_path, text_dim, img_dim, proj_dim=256, test_dataloader=None):
    """
    测试 MultimodalAddProjectionModel 模型
    
    Args:
        model_path: 模型权重文件路径
        text_dim: 文本特征维度
        img_dim: 图像特征维度
        proj_dim: 投影维度，默认256
        test_dataloader: 测试数据加载器，如果为None则使用val_dl
    """
    y_true = []
    y_pred = []
    user_ids_list = []
    movie_ids_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    state_dict = torch.load(model_path, map_location=device)
    model = MultimodalAddProjectionModel(
        text_input_dim=text_dim,
        img_input_dim=img_dim,
        projection_dim=proj_dim
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 使用指定的dataloader或默认的val_dl
    dataloader = test_dataloader if test_dataloader is not None else val_dl
    
    print(f"Testing model: {model_path}")
    print(f"Using device: {device}")
    print(f"Model config: text_dim={text_dim}, img_dim={img_dim}, proj_dim={proj_dim}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            text_features = batch["text_features"].to(device)
            img_features = batch["img_features"].to(device)
            outputs = model(text_features, img_features, return_emb=False)
            y_true.extend(batch["label"].cpu().numpy())
            y_pred.extend(outputs.cpu().squeeze().numpy())
            user_ids_list.extend(batch["user_id"].cpu().numpy())
            movie_ids_list.extend(batch["movie_id"].cpu().numpy())
    
    # 计算指标
    auc = roc_auc_score(y_true, y_pred)
    rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
    
    print(f"\n=== Test Results ===")
    print(f"AUC: {auc:.4f}")
    print(f"Precision@5: {rec_metrics['Precision@5']:.4f} | Precision@10: {rec_metrics['Precision@10']:.4f} | Precision@20: {rec_metrics['Precision@20']:.4f}")
    print(f"Recall@5: {rec_metrics['Recall@5']:.4f} | Recall@10: {rec_metrics['Recall@10']:.4f} | Recall@20: {rec_metrics['Recall@20']:.4f}")
    print(f"NDCG@5: {rec_metrics['NDCG@5']:.4f} | NDCG@10: {rec_metrics['NDCG@10']:.4f} | NDCG@20: {rec_metrics['NDCG@20']:.4f}")
    print(f"MAP: {rec_metrics['MAP']:.4f}")
    
    return {'auc': auc, **rec_metrics}, y_true, y_pred



def test_model(model_path, test_dataloader=None, mode="multimodal"):
    """
    测试 MultimodalRecModel 模型
    
    Args:
        model_path: 模型权重文件路径
        test_dataloader: 测试数据加载器，如果为None则使用val_dl
        mode: 模式 ("multimodal", "text", "image")
    """
    y_true = []
    y_pred = []
    user_ids_list = []
    movie_ids_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    
    # 推断输入维度
    sample_dl = test_dataloader if test_dataloader is not None else val_dl
    sample = next(iter(sample_dl))
    if mode == "text":
        input_dim = sample["text_features"].shape[1]
    elif mode == "image":
        input_dim = sample["img_features"].shape[1]
    else:
        input_dim = sample["text_features"].shape[1] + sample["img_features"].shape[1]
    
    model = MultimodalRecModel(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    dataloader = test_dataloader if test_dataloader is not None else val_dl
    
    print(f"Testing model: {model_path}")
    print(f"Mode: {mode}, input_dim: {input_dim}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs = _build_inputs(batch, device, mode)
            outputs = model(inputs)
            y_true.extend(batch["label"].cpu().numpy())
            y_pred.extend(outputs.cpu().squeeze().numpy())
            user_ids_list.extend(batch["user_id"].cpu().numpy())
            movie_ids_list.extend(batch["movie_id"].cpu().numpy())
    
    # 计算指标
    auc = roc_auc_score(y_true, y_pred)
    rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])
    
    print(f"\n=== Test Results ===")
    print(f"AUC: {auc:.4f}")
    print(f"Precision@5: {rec_metrics['Precision@5']:.4f} | Precision@10: {rec_metrics['Precision@10']:.4f} | Precision@20: {rec_metrics['Precision@20']:.4f}")
    print(f"Recall@5: {rec_metrics['Recall@5']:.4f} | Recall@10: {rec_metrics['Recall@10']:.4f} | Recall@20: {rec_metrics['Recall@20']:.4f}")
    print(f"NDCG@5: {rec_metrics['NDCG@5']:.4f} | NDCG@10: {rec_metrics['NDCG@10']:.4f} | NDCG@20: {rec_metrics['NDCG@20']:.4f}")
    print(f"MAP: {rec_metrics['MAP']:.4f}")
    
    return {'auc': auc, **rec_metrics}, y_true, y_pred



def precision_at_k(y_true, y_pred, k=10):
    """计算Precision@K"""
    if len(y_true) == 0:
        return 0.0
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    relevant = sum(y_true[i] for i in top_k_indices)
    return relevant / min(k, len(y_true))

def recall_at_k(y_true, y_pred, k=10):
    """计算Recall@K"""
    if len(y_true) == 0 or sum(y_true) == 0:
        return 0.0
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    relevant = sum(y_true[i] for i in top_k_indices)
    return relevant / sum(y_true)

def average_precision(y_true, y_pred):
    """计算Average Precision (AP)"""
    if sum(y_true) == 0:
        return 0.0
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    precisions = []
    num_relevant = 0
    for i, label in enumerate(y_true_sorted):
        if label > 0:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    return sum(precisions) / sum(y_true) if sum(y_true) > 0 else 0.0

def ndcg_at_k(y_true, y_pred, k=10):
    """计算NDCG@K"""
    if len(y_true) == 0:
        return 0.0
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    dcg = sum(y_true[idx] / np.log2(i + 2) for i, idx in enumerate(top_k_indices))
    ideal_sorted = sorted(y_true, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    return dcg / idcg if idcg > 0 else 0.0

def compute_recommendation_metrics(user_ids, movie_ids, y_true, y_pred, ks=[5, 10, 20]):
    """按用户分组计算推荐系统指标"""
    user_data = defaultdict(lambda: {'movie_ids': [], 'labels': [], 'scores': []})
    
    for uid, mid, label, score in zip(user_ids, movie_ids, y_true, y_pred):
        user_data[uid]['movie_ids'].append(mid)
        user_data[uid]['labels'].append(label)
        user_data[uid]['scores'].append(score)
    
    metrics = {}
    for k in ks:
        metrics[f'Precision@{k}'] = []
        metrics[f'Recall@{k}'] = []
        metrics[f'NDCG@{k}'] = []
    metrics['MAP'] = []
    
    for uid, data in user_data.items():
        labels = np.array(data['labels'])
        scores = np.array(data['scores'])
        
        if sum(labels) == 0:  # 跳过没有正样本的用户
            continue
            
        for k in ks:
            metrics[f'Precision@{k}'].append(precision_at_k(labels, scores, k))
            metrics[f'Recall@{k}'].append(recall_at_k(labels, scores, k))
            metrics[f'NDCG@{k}'].append(ndcg_at_k(labels, scores, k))
        metrics['MAP'].append(average_precision(labels, scores))
    
    # 计算平均值
    result = {}
    for key, values in metrics.items():
        result[key] = np.mean(values) if len(values) > 0 else 0.0
    return result


def _build_inputs(batch, device, mode: str):
    text_features = batch["text_features"].to(device)
    img_features = batch["img_features"].to(device)
    if mode == "text":
        return text_features
    if mode == "image":
        return img_features
    # default multimodal
    return torch.cat((text_features, img_features), dim=-1)


def _get_logger(mode: str) -> logging.Logger:
    logs_dir = pathlib.Path("logs") / mode
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_{timestamp}.log"

    logger = logging.getLogger(f"trainer.{mode}.{timestamp}")
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


def train_and_validate(train_dl, val_dl, input_dim, epoch_num=10, lr=1e-3, mode: str = "multimodal"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = _get_logger(mode)
    logger.info(f"Using device: {device}")
    logger.info(f"Mode: {mode} | input_dim: {input_dim} | epochs: {epoch_num} | lr: {lr}")

    model = MultimodalRecModel(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    model_output_dir = pathlib.Path("model")
    model_output_dir.mkdir(exist_ok=True)

    for epoch in range(1, epoch_num + 1):
        # --------------------
        # 🔹 Train
        # --------------------
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            labels = batch["label"].to(device).unsqueeze(1)
            inputs = _build_inputs(batch, device, mode)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)

        # --------------------
        # 🔹 Validation
        # --------------------
        model.eval()
        y_true, y_pred = [], []
        user_ids_list, movie_ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Validating Epoch {epoch}"):
                labels = batch["label"].cpu().numpy()
                inputs = _build_inputs(batch, device, mode)
                outputs = model(inputs)
                preds = outputs.cpu().squeeze().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)
                user_ids_list.extend(batch["user_id"].cpu().numpy())
                movie_ids_list.extend(batch["movie_id"].cpu().numpy())

        # 计算 AUC
        auc = roc_auc_score(y_true, y_pred)
        
        # 计算推荐系统指标
        rec_metrics = compute_recommendation_metrics(user_ids_list, movie_ids_list, y_true, y_pred, ks=[5, 10, 20])

        # --------------------
        # 🔹 打印与保存
        # --------------------
        metrics_str = f"AUC: {auc:.4f} | P@5: {rec_metrics['Precision@5']:.4f} | R@5: {rec_metrics['Recall@5']:.4f} | NDCG@5: {rec_metrics['NDCG@5']:.4f} | MAP: {rec_metrics['MAP']:.4f}"
        logger.info(f"Epoch {epoch:02d} | Mode: {mode} | Train Loss: {avg_train_loss:.4f} | Val {metrics_str} | Best AUC: {best_auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            suffix = {"multimodal": "text&img_model.pth", "text": "best_model_text.pth", "image": "best_model_image.pth"}[mode]
            torch.save(model.state_dict(), model_output_dir / suffix)
            logger.info(f"✅ New best model saved ({mode}) with AUC={best_auc:.4f}")

    logger.info(f"Training finished. Mode: {mode} | Best AUC={best_auc:.4f}")

def _infer_input_dim(dataloader, mode: str) -> int:
    sample = next(iter(dataloader))
    if mode == "text":
        return sample["text_features"].shape[1]
    if mode == "image":
        return sample["img_features"].shape[1]
    return sample["text_features"].shape[1] + sample["img_features"].shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recommender with different modalities")
    parser.add_argument("--model", choices=["baseline", "projection", "projectiongate", "transformer", "average_fusion","heavy_transformer"], default="baseline", 
                        help="Model type: 'baseline' for MultimodalRecModel, 'projection' for MultimodalAddProjectionModel, 'projectiongate' for MultimodalAddProjectionGateModel, 'transformer' for MultimodalTransformerModel")
    parser.add_argument("--mode", choices=["multimodal", "text", "image"], default="multimodal",
                        help="Modality mode (only for baseline model)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default 5e-4 for projection/projectiongate/transformer, 1e-3 for baseline)")
    # Projection model specific parameters
    parser.add_argument("--proj-dim", type=int, default=256, help="Projection dimension for projection/projectiongate model")
    parser.add_argument("--lambda-align", type=float, default=0.05, help="Weight for contrastive alignment loss (default 0.05)")
    # Transformer model specific parameters
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for transformer model (default 256)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads for transformer (default 8)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers (default 2)")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable learning rate scheduler")
    args = parser.parse_args()

    if args.model == "projection":
        # 训练投影模型（带对比损失）
        sample = next(iter(train_dl))
        text_dim = sample["text_features"].shape[1]
        img_dim = sample["img_features"].shape[1]
        print(f"Training MultimodalAddProjectionModel")
        print(f"text_dim: {text_dim}, img_dim: {img_dim}, proj_dim: {args.proj_dim}, lambda_align: {args.lambda_align}")
        
        train_MultimodalAddProjectionModel(
            train_dl=train_dl,
            val_dl=val_dl,
            text_dim=text_dim,
            img_dim=img_dim,
            proj_dim=args.proj_dim,
            lambda_align=args.lambda_align,
            epoch_num=args.epochs,
            lr=args.lr,
            use_scheduler=not args.no_scheduler
        )
    elif args.model == "projectiongate":
        # 训练投影Gate模型（带Gate机制和对比损失）
        sample = next(iter(train_dl))
        text_dim = sample["text_features"].shape[1]
        img_dim = sample["img_features"].shape[1]
        print(f"Training MultimodalAddProjectionGateModel")
        print(f"text_dim: {text_dim}, img_dim: {img_dim}, proj_dim: {args.proj_dim}, lambda_align: {args.lambda_align}")
        
        train_MultimodalProjectionGateModel(
            train_dl=train_dl,
            val_dl=val_dl,
            text_dim=text_dim,
            img_dim=img_dim,
            proj_dim=args.proj_dim,
            lambda_align=args.lambda_align,
            epoch_num=args.epochs,
            lr=args.lr,
            use_scheduler=not args.no_scheduler
        )
    elif args.model == "transformer":
        # 训练Transformer模型
        sample = next(iter(train_dl))
        text_dim = sample["text_features"].shape[1]
        img_dim = sample["img_features"].shape[1]
        print(f"Training MultimodalTransformerModel")
        print(f"text_dim: {text_dim}, img_dim: {img_dim}, hidden_dim: {args.hidden_dim}, num_heads: {args.num_heads}, num_layers: {args.num_layers}, lambda_align: {args.lambda_align}")
        
        train_MultimodalTransformerModel(
            train_dl=train_dl,
            val_dl=val_dl,
            text_dim=text_dim,
            img_dim=img_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            lambda_align=args.lambda_align,
            epoch_num=args.epochs,
            lr=args.lr,
            use_scheduler=not args.no_scheduler
        )
    elif args.model == "average_fusion":
        # 训练average_fusion模型
        sample = next(iter(train_dl))
        text_dim = sample["text_features"].shape[1]
        img_dim = sample["img_features"].shape[1]
        print(f"Training average_fusion_model")
        print(f"text_dim: {text_dim}, img_dim: {img_dim}")
        train_average_fusion_model(
            train_dl=train_dl,
            val_dl=val_dl,
            text_dim=text_dim,
            img_dim=img_dim,
            epoch_num=args.epochs,
            lr=args.lr,
            use_scheduler=not args.no_scheduler
        )
    elif args.model == "heavy_transformer":
        # 训练heavy_transformer模型
        sample = next(iter(train_dl))
        text_dim = sample["text_features"].shape[1]
        img_dim = sample["img_features"].shape[1]
        print(f"Training HeavyTransformerModel")
        print(f"text_dim: {text_dim}, img_dim: {img_dim}, hidden_dim: {args.hidden_dim}, num_heads: {args.num_heads}, num_layers: {args.num_layers}")
        train_HeavyTransformerModel(
            train_dl=train_dl,
            val_dl=val_dl,
            text_dim=text_dim,
            img_dim=img_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            epoch_num=args.epochs,
            lr=args.lr,
            use_scheduler=not args.no_scheduler
        )
    else:
        # 训练baseline模型
        mode = args.mode
        input_dim = _infer_input_dim(train_dl, mode)
        # baseline模型默认使用1e-3，如果用户没有指定lr则使用默认值
        baseline_lr = args.lr if args.lr != 5e-4 else 1e-3
        print(f"Mode: {mode}, input_dim: {input_dim}")

        train_and_validate(
            train_dl=train_dl,
            val_dl=val_dl,
            input_dim=input_dim,
            epoch_num=args.epochs,
            lr=baseline_lr,
            mode=mode
        )