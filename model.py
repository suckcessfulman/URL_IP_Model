import torch
import torch.nn as nn
from transformers import DistilBertModel
import torch.nn.functional as F
import math

class BiModalAttention(nn.Module):
    """集成Cross-Modal BERT完整注意力机制"""

    def __init__(self, hidden_size=128, num_heads=8, temperature=0.2):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.temperature = temperature

        # 多头部投影
        self.proj_t = nn.Linear(hidden_size, hidden_size)
        self.proj_i = nn.Linear(hidden_size, hidden_size)

        # 来自Cross-Modal BERT的缩放因子
        self.scale = 1.0 / math.sqrt(self.d_k)

        # 保持原有层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_feat, ip_feat):
        # 多头部分割
        def _reshape(x):
            return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        # 投影变换
        projected_text = self.proj_t(text_feat)
        projected_ip = self.proj_i(ip_feat)

        # 多头处理
        q_text = _reshape(projected_text)
        k_ip = _reshape(projected_ip)
        v_ip = _reshape(projected_ip)

        q_ip = _reshape(projected_ip)
        k_text = _reshape(projected_text)
        v_text = _reshape(projected_text)

        # 带缩放的注意力计算
        attn_text = (q_text @ k_ip.transpose(-2, -1)) * self.scale
        attn_text = attn_text.softmax(dim=-1) / self.temperature
        o_text = (attn_text @ v_ip).transpose(1, 2).contiguous().flatten(2)

        attn_ip = (q_ip @ k_text.transpose(-2, -1)) * self.scale
        attn_ip = attn_ip.softmax(dim=-1) / self.temperature
        o_ip = (attn_ip @ v_text).transpose(1, 2).contiguous().flatten(2)

        # 保持原有融合方式
        a_text = self.layer_norm(torch.mul(o_text, projected_text))
        a_ip = self.layer_norm(torch.mul(o_ip, projected_ip))

        return a_text, a_ip

class ContrastiveLearning(nn.Module):
    """集成多模态NCE损失"""

    def __init__(self, temperature=0.1, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 模态内负样本权重

    def forward(self, url_emb, ip_emb):
        # 归一化处理
        url_emb = F.normalize(url_emb, p=2, dim=1)
        ip_emb = F.normalize(ip_emb, p=2, dim=1)
        batch_size = url_emb.size(0)

        # 跨模态相似度
        cross_sim = (url_emb @ ip_emb.T) / self.temperature

        # 模态内相似度（来自Cross-Modal BERT）
        intra_url_sim = (url_emb @ url_emb.T) / self.temperature
        intra_ip_sim = (ip_emb @ ip_emb.T) / self.temperature

        # 构建联合相似度矩阵
        sim_matrix = torch.cat([
            torch.cat([intra_url_sim, cross_sim], dim=1),
            torch.cat([cross_sim.T, intra_ip_sim], dim=1)
        ], dim=0)

        # 创建标签mask
        labels = torch.arange(batch_size, device=url_emb.device)
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=url_emb.device)
        mask[labels, labels + batch_size] = 1  # 跨模态正样本
        mask[labels + batch_size, labels] = 1

        # 排除自相似度
        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size, device=url_emb.device)

        # 计算对比损失
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        loss = -(mask * log_prob).sum() / mask.sum()
        return loss


class URL_IP_Model(nn.Module):
    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        # BERT编码器
        self.bert = DistilBertModel.from_pretrained(bert_path)

        # URL处理分支（保持原有结构）
        self.url_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 增强IP编码器
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256),
            nn.GELU(),  # 更平滑的激活函数
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        # 使用改进的注意力模块
        self.bi_attn = BiModalAttention(hidden_size=128, num_heads=num_heads)

        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 2) #2分类就为2
        )

        # 使用改进的对比学习
        self.contrastive = ContrastiveLearning(temperature=0.1, alpha=0.3)

    def forward(self, input_ids, attention_mask, ip_data, labels, current_epoch=None, total_epochs=None):
        # 修正BERT调用方式
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 保持原有特征提取方式
        url_feature = self.url_fc(bert_output.last_hidden_state[:, 0, :])  # [batch_size, 128]
        ip_feature = self.ip_encoder(ip_data)

        # 改进的注意力交互
        attn_url, attn_ip = self.bi_attn(url_feature.unsqueeze(1), ip_feature.unsqueeze(1))
        attn_url = attn_url.squeeze(1)
        attn_ip = attn_ip.squeeze(1)

        # 层次化融合
        combined = torch.cat([
            url_feature + 0.7 * attn_url,  # 可调节的注意力权重
            ip_feature + 0.7 * attn_ip
        ], dim=1)

        # 分类预测
        logits = self.classifier(combined)

        # 损失计算
        contrastive_loss = self.contrastive(url_feature, ip_feature)
        cls_loss = F.cross_entropy(logits, labels)
        total_loss = cls_loss + 0.3 * contrastive_loss

        return total_loss, logits