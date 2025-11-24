import torch.nn as nn
import torch.nn.functional as F
import torch
class ProjecitonLayer(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,dropout_rate = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self,features):
        return self.net(features)


def cosine_sim(a,b,eps = 1e-8):
    norm_a = F.normalize(a,dim=-1,eps=eps)
    norm_b = F.normalize(b,dim=-1,eps=eps)
    return (norm_a * norm_b).sum(dim=-1)

class SemanticGate(nn.Module):
    """
    Gate f(t, i) -> scalar or vector in [0,1]
    Input: t (B,D), i (B,D)
    Output: g (B, D) if vector_gate else (B,1)
    """
    def __init__(self,proj_dim,gate_hidden_dim,vector_gate = False):
        super().__init__()
        self.vector_gate = vector_gate
        in_dim = proj_dim*3 +1 # t i t*i ,sim
        out_dim = proj_dim if vector_gate else 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim,gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim,out_dim),
        )
    def forward(self,t,i):
        sim = cosine_sim(t,i).unsqueeze(-1)
        inter = t*i
        concat = torch.cat([t,i,inter,sim],dim=-1)
        g = self.mlp(concat)
        return torch.sigmoid(g)


class CGFMRecommender(nn.Module):
    """
    Final model for recommendation:
    text and image features are projected to a common space,
    semantic gate -> compute g
    fused item vector = 
    """
    def __init__(self, text_input_dim, img_input_dim,num_users,num_items,temperature=0.07,user_emb_dim = 64, proj_hidden_dim = 512, proj_dim=512, vector_gate=False,user_proj_dim=64):
        super().__init__()
        self.text_proj = ProjecitonLayer(text_input_dim,proj_hidden_dim,proj_dim)
        self.img_proj = ProjecitonLayer(img_input_dim,proj_hidden_dim,proj_dim)
        self.gate = SemanticGate(proj_dim,gate_hidden_dim=256,vector_gate=vector_gate)
        self.user_emb = nn.Embedding(num_users,user_emb_dim)
        self.item_emb = nn.Embedding(num_items,proj_dim)
        ####### optianal user porjection layer -> proj_dim #######
        if user_emb_dim != proj_dim:
            self.user_proj = nn.Linear(user_emb_dim,proj_dim)
        else: 
            self.user_proj = None
            
        self.temperature = temperature
        ###
    
    def forward_item(self,img_features,text_features,item_ids=None):
        text = self.text_proj(text_features)
        img = self.img_proj(img_features)
        g = self.gate(text,img)
        fused = g * text + (1 - g) * img
        if item_ids is not None:
            fused = fused +self.item_id_emb(item_ids)
        return text,img,fused,g
    
    def score(self,user_ids,item_rep):
        u = self.user_emb(user_ids)
        if self.user_proj is not None:
            u = self.user_proj(u)
        scores = (u * item_rep).sum(dim=-1)
        return scores
    
    def forward(self,user_ids,item_ids,img_features,text_features):
        text,img,fused,g = self.forward_item(img_features,text_features,item_ids)
        scores = self.score(user_ids,fused)
        return scores,text,img,fused,g
    
    def contrastive_logits(self,text_rep,img_rep):
        t_n = F.normalize(text_rep,dim=-1)
        i_n = F.normalize(img_rep,dim=-1)
        logits = torch.matmul(t_n,i_n.t()) / self.temperature
        return logits
    
    def info_nce_loss(logits):
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss_t2i = F.cross_entropy(logits,labels)
        loss_i2t = F.cross_entropy(logits.t(),labels)
        return (loss_t2i + loss_i2t) / 2
    
    def bpr_loss(pos_scores, neg_scores):
        diff = pos_scores - neg_scores
        return -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    
    

def train_contrtrastive():
    model.train()
    text_rep = model.text_proj(text_features)
    pos_logits, t_pos, i_pos, fused_pos, g_pos = model(user_ids, pos_item, pos_text, pos_img)
    with torch.no_grad():
        t_neg, i_neg, fused_neg, g_neg = model.forward_item_rep(neg_text, neg_img, neg_item)
        neg_logits = model.score(user, fused_neg)

        rec_loss = bpr_loss(pos_logits, neg_logits)
        contra_logits = model.contrastive_logits(t_pos, i_pos)
        contr_loss = info_nce_loss(contra_logits)
        loss = rec_loss + lambda_contrast * contr_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * user.size(0)
    