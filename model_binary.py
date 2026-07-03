import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel


class BiModalAttention(nn.Module):
    """Bi-modal multi-head cross attention between URL and IP features."""

    def __init__(self, hidden_size=128, num_heads=8, temperature=0.2):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.temperature = temperature
        self.dropout = nn.Dropout(0.2)
        self.proj_t = nn.Linear(hidden_size, hidden_size)
        self.proj_i = nn.Linear(hidden_size, hidden_size)
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_feat, ip_feat):
        def _reshape(x):
            return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        projected_text = self.proj_t(text_feat)
        projected_ip = self.proj_i(ip_feat)

        q_text = _reshape(projected_text)
        k_ip = _reshape(projected_ip)
        v_ip = _reshape(projected_ip)
        q_ip = _reshape(projected_ip)
        k_text = _reshape(projected_text)
        v_text = _reshape(projected_text)

        attn_text = (q_text @ k_ip.transpose(-2, -1)) * self.scale
        attn_text = attn_text.softmax(dim=-1) / self.temperature
        attn_text = self.dropout(attn_text)
        o_text = (attn_text @ v_ip).transpose(1, 2).contiguous().flatten(2)

        attn_ip = (q_ip @ k_text.transpose(-2, -1)) * self.scale
        attn_ip = attn_ip.softmax(dim=-1) / self.temperature
        attn_ip = self.dropout(attn_ip)
        o_ip = (attn_ip @ v_text).transpose(1, 2).contiguous().flatten(2)

        a_text = self.layer_norm(torch.mul(o_text, projected_text))
        a_ip = self.layer_norm(torch.mul(o_ip, projected_ip))
        return a_text, a_ip


class ContrastiveLearning(nn.Module):
    """Symmetric URL-IP contrastive learning loss."""

    def __init__(self, temperature=0.1, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, url_emb, ip_emb):
        url_emb = F.normalize(url_emb, p=2, dim=1)
        ip_emb = F.normalize(ip_emb, p=2, dim=1)
        batch_size = url_emb.size(0)

        cross_sim = (url_emb @ ip_emb.T) / self.temperature
        intra_url_sim = (url_emb @ url_emb.T) / self.temperature
        intra_ip_sim = (ip_emb @ ip_emb.T) / self.temperature

        sim_matrix = torch.cat([
            torch.cat([intra_url_sim, cross_sim], dim=1),
            torch.cat([cross_sim.T, intra_ip_sim], dim=1),
        ], dim=0)

        labels = torch.arange(batch_size, device=url_emb.device)
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=url_emb.device)
        mask[labels, labels + batch_size] = 1
        mask[labels + batch_size, labels] = 1

        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size, device=url_emb.device)
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
        return -(mask * log_prob).sum() / mask.sum().clamp_min(1.0)


class URL_IP_Model(nn.Module):
    """URLChecker-style URL-IP model with dynamic gated fusion."""

    NUM_CLASSES = 2

    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_path)
        self.url_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.bi_attn = BiModalAttention(hidden_size=128, num_heads=num_heads)
        self.fusion_gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.Dropout(0.2), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, self.NUM_CLASSES),
        )
        self.contrastive = ContrastiveLearning(temperature=0.1, alpha=0.3)

    def forward(self, input_ids, attention_mask, ip_data, labels=None, current_epoch=None, total_epochs=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        url_feature = self.url_fc(bert_output.last_hidden_state[:, 0, :])
        ip_feature = self.ip_encoder(ip_data)

        attn_url, attn_ip = self.bi_attn(url_feature.unsqueeze(1), ip_feature.unsqueeze(1))
        attn_url = attn_url.squeeze(1)
        attn_ip = attn_ip.squeeze(1)

        gate_weight_url = self.fusion_gate(torch.cat([url_feature, attn_url], dim=-1))
        fused_url = gate_weight_url * url_feature + (1.0 - gate_weight_url) * attn_url

        gate_weight_ip = self.fusion_gate(torch.cat([ip_feature, attn_ip], dim=-1))
        fused_ip = gate_weight_ip * ip_feature + (1.0 - gate_weight_ip) * attn_ip

        logits = self.classifier(torch.cat([fused_url, fused_ip], dim=1))
        if labels is None:
            return logits

        contrastive_loss = self.contrastive(url_feature, ip_feature)
        cls_loss = F.cross_entropy(logits, labels)
        if current_epoch is not None and total_epochs is not None and total_epochs > 0:
            contrastive_weight = 0.3 * (1.0 - float(current_epoch) / float(total_epochs))
        else:
            contrastive_weight = 0.3
        return cls_loss + contrastive_weight * contrastive_loss, logits
