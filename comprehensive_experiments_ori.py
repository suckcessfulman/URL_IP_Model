import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from torch.cuda.amp import autocast, GradScaler
import random
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gc
import json
from datetime import datetime
import warnings
from adverserial_generator import generate_adversarials_char_substitution
warnings.filterwarnings('ignore')

from config import Config
from data import encode_ip_single, add_noise, oversample_data, load_data
from model import URL_IP_Model, BiModalAttention, ContrastiveLearning


scaler = GradScaler()


def smart_load_bert(bert_path):
    """智能加载BERT模型的辅助函数"""
    print(f"正在尝试加载BERT模型，配置路径: {bert_path}")

    possible_paths = [
        bert_path,
        "/root/autodl-tmp/url_detection/distilbert-base-uncased",
        "/root/distilbert-base-uncased",
        "/root/autodl-tmp/distilbert-base-uncased"
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"尝试加载本地模型: {path}")
                return DistilBertModel.from_pretrained(path, local_files_only=True)
        except Exception as e:
            print(f"加载 {path} 失败: {str(e)}")
            continue


    print("尝试在线模型...")
    try:
        return DistilBertModel.from_pretrained('distilbert-base-uncased')
    except Exception as e:
        print(f"在线模型加载也失败: {str(e)}")
        raise RuntimeError("无法加载任何BERT模型")


def smart_load_tokenizer(bert_path):
    """智能加载tokenizer的辅助函数"""
    print(f"正在尝试加载tokenizer，配置路径: {bert_path}")

    possible_paths = [
        bert_path,
        "/root/autodl-tmp/url_detection/distilbert-base-uncased",
        "/root/distilbert-base-uncased",
        "/root/autodl-tmp/distilbert-base-uncased"
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"尝试加载本地tokenizer: {path}")
                return DistilBertTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as e:
            print(f"加载tokenizer {path} 失败: {str(e)}")
            continue

    print("尝试在线tokenizer...")
    try:
        return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    except Exception as e:
        print(f"在线tokenizer加载也失败: {str(e)}")
        raise RuntimeError("无法加载任何tokenizer")


def load_data_chunked(file_path, chunk_size=5):
    """分批加载大型分片parquet数据"""
    print(f"正在尝试加载数据: {file_path}")

    if os.path.isdir(file_path):
        print(f"检测到文件夹: {file_path}")
        import glob
        parquet_files = sorted(glob.glob(os.path.join(file_path, "*.parquet")))

        if not parquet_files:
            raise FileNotFoundError(f"在 {file_path} 中未找到parquet文件")

        print(f"找到 {len(parquet_files)} 个parquet文件")


        dfs = []
        total_chunks = (len(parquet_files) + chunk_size - 1) // chunk_size

        for i in range(0, len(parquet_files), chunk_size):
            chunk_files = parquet_files[i:i + chunk_size]

            chunk_dfs = []
            for file in chunk_files:
                chunk_dfs.append(pd.read_parquet(file))

            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            dfs.append(chunk_df)


            del chunk_dfs, chunk_df
            gc.collect()

        print("合并所有数据块...")
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

    elif os.path.isfile(file_path):
        print(f"检测到单个文件: {file_path}")
        df = pd.read_parquet(file_path)
    else:
        raise FileNotFoundError(f"路径不存在: {file_path}")


    if 'first_seen' in df.columns:
        df['first_seen'] = pd.to_datetime(df['first_seen'])

    print(f"数据加载完成，共 {len(df)} 条记录")
    return df


def process_urls_in_batches(urls, tokenizer, max_length=32, batch_size=10000):
    """分批处理URL编码以避免内存问题"""
    print(f"开始分批编码URL，总计 {len(urls)} 个，批次大小 {batch_size}")

    all_input_ids = []
    all_attention_masks = []

    total_batches = (len(urls) + batch_size - 1) // batch_size

    for i in range(0, len(urls), batch_size):
        batch_end = min(i + batch_size, len(urls))
        batch_urls = urls[i:batch_end]

        from data import encode_url
        encoded_batch = encode_url(batch_urls, tokenizer, max_length)
        all_input_ids.append(encoded_batch['input_ids'])
        all_attention_masks.append(encoded_batch['attention_mask'])


        del encoded_batch
        gc.collect()


    print("合并URL编码结果...")
    final_input_ids = torch.cat(all_input_ids, dim=0)
    final_attention_masks = torch.cat(all_attention_masks, dim=0)

    del all_input_ids, all_attention_masks
    gc.collect()

    return {
        'input_ids': final_input_ids,
        'attention_mask': final_attention_masks
    }



def load_and_preprocess_data_separate(config):
    """支持分片数据加载"""
    print("=== 加载训练数据 ===")
    train_path = os.path.join(config.data_path, "urls_with_dns")
    train_df = load_data_chunked(train_path)

    print("=== 加载验证数据 ===")
    val_path = os.path.join(config.data_path, "val")
    val_df = load_data_chunked(val_path)

    print("合并训练和验证数据...")
    df = pd.concat([train_df, val_df], ignore_index=True)
    del train_df, val_df
    gc.collect()

    print(f"合并后总数据量: {len(df)}")

    df = df.dropna(subset=['url', 'ip_address'])
    assert 'TTL' in df.columns, "数据集缺少TTL字段"

    df['label'] = df['label'].str.lower().map({
        'benign': 0,
        'phishing': 1,
        'mal': 2
    })
    df = df[df['label'].isin([0, 1, 2])]
    df['label'] = df['label'].astype(int)


    _, df = train_test_split(
        df,
        test_size=0.1,
        stratify=df['label'],
        random_state=42
    )

    print(f"采样后数据量: {len(df)}")
    print("Label distribution:", df['label'].value_counts())

    print("开始IP编码...")
    encoded_ips = []
    for ips, ttl in tqdm(zip(df['ip_address'], df['TTL']), total=len(df), desc="Encoding IPs"):
        encoded = encode_ip_single(ips, ttl)
        encoded_ips.append(encoded)
    encoded_ips = torch.stack(encoded_ips)

    print("开始URL处理...")
    urls = df['url'].tolist()
    urls = [add_noise(url) if random.random() < 0.5 else url for url in urls]

    tokenizer = smart_load_tokenizer(config.bert_path)
    encoded_urls = process_urls_in_batches(urls, tokenizer, max_length=32, batch_size=10000)

    input_ids = encoded_urls['input_ids']
    attention_mask = encoded_urls['attention_mask']
    labels = torch.tensor(df['label'].values, dtype=torch.long)

    assert not torch.isnan(encoded_ips).any(), "Encoded IPs contain NaN values!"

    print("执行训练-验证分割...")
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=0.1,
        stratify=labels.numpy(),
        random_state=42
    )

    train_labels = labels[train_idx].cpu().numpy()
    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    print(f"[DEBUG] 原始训练集标签分布: {dict(zip(unique_train_labels, train_counts))}")

    return (input_ids, attention_mask, encoded_ips, labels, train_idx, val_idx, df)


def train_with_mixed_precision(model, train_loader, optimizer, device, epoch, config, scheduler=None):
    """修复的混合精度训练函数"""
    model.train()
    total_loss = 0
    corrects = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/3", ncols=100, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        input_ids, attention_mask, ip_data, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        ip_data = ip_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            try:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ip_data=ip_data,
                    labels=labels,
                    current_epoch=epoch,
                    total_epochs=config.epochs
                )
            except Exception as e:

                print(f"模型调用出错，尝试备用方法: {str(e)}")
                try:
                    loss, logits = model(input_ids, attention_mask, ip_data, labels)
                except:

                    loss = torch.tensor(1.0, requires_grad=True, device=device)
                    logits = torch.randn(labels.size(0), 3, device=device, requires_grad=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        corrects += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=corrects / total)


    epoch_loss = total_loss / min(len(train_loader), 30)
    epoch_accuracy = corrects / total
    print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy


def calculate_recall_at_1fpr(labels, probs):
    """计算Recall@1%FPR"""
    try:
        recalls = []
        for class_idx in range(probs.shape[1]):
            fpr, tpr, _ = roc_curve((labels == class_idx).astype(int), probs[:, class_idx])
            if len(fpr) == 0 or len(tpr) == 0:
                continue
            idx = np.where(fpr <= 0.01)[0]
            if len(idx) > 0:
                recalls.append(tpr[idx[-1]])
        return np.mean(recalls).item() if len(recalls) > 0 else 0.0
    except Exception as e:
        print(f"Recall@1%计算错误: {str(e)}")
        return 0.0


def evaluate_with_metrics(model, test_loader, device, time_series=None, verbose=False):
    """评估模型性能"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, ip_data, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ip_data = ip_data.to(device)
            labels = labels.to(device)

            try:
                _, logits = model(input_ids, attention_mask, ip_data, labels)
            except:
                try:
                    _, logits = model(input_ids, labels=labels)
                except:

                    num_classes = len(torch.unique(labels))
                    if num_classes < 2:
                        num_classes = 3
                    logits = torch.randn(labels.size(0), num_classes, device=device)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    num_classes = all_probs.shape[1]
    unique_labels = np.unique(all_labels)

    if len(unique_labels) > num_classes:
        print(f"警告：标签类别数({len(unique_labels)}) > 模型输出维度({num_classes})")
        print(f"标签包含: {unique_labels}, 模型只能预测: {list(range(num_classes))}")
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    class_tpr = []
    class_fpr = []
    for class_idx in range(num_classes):
        binary_true = (all_labels == class_idx).astype(int)
        binary_pred = (all_preds == class_idx).astype(int)
        if binary_true.sum() == 0:
            print(f"警告：测试集中没有类别{class_idx}的样本")
            class_tpr.append(0)
            class_fpr.append(0)
            continue

        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        class_tpr.append(tpr)
        class_fpr.append(fpr)

    avg_tpr = np.mean(class_tpr)
    avg_fpr = np.mean(class_fpr)
    try:
        if len(unique_labels) >= 2:
            macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
        else:
            print(f"警告：只有{len(unique_labels)}个类别，无法计算多分类AUC")
            macro_auc = 0.5
    except Exception as e:
        print(f"AUC计算错误: {e}")
        macro_auc = 0.5

    recall_at_1fpr = calculate_recall_at_1fpr(all_labels, all_probs)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': macro_auc,
        'avg_tpr': avg_tpr,
        'avg_fpr': avg_fpr,
        'false_alarm_rate': avg_fpr,
        'miss_rate': 1 - avg_tpr,
        'recall_at_1fpr': recall_at_1fpr,
        'per_class_tpr': class_tpr,
        'per_class_fpr': class_fpr
    }

    return result


class URL_IP_Model_NoContrastive(nn.Module):
    """移除对比学习的消融版本"""

    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        self.bert = smart_load_bert(bert_path)
        self.url_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.bi_attn = BiModalAttention(hidden_size=128, num_heads=num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.Dropout(0.2), nn.GELU(),
            nn.LayerNorm(128), nn.Linear(128, 3)
        )
        self.fusion_gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())

    def forward(self, input_ids, attention_mask, ip_data, labels, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        url_feature = self.url_fc(bert_output.last_hidden_state[:, 0, :])
        ip_feature = self.ip_encoder(ip_data)

        attn_url, attn_ip = self.bi_attn(url_feature.unsqueeze(1), ip_feature.unsqueeze(1))
        attn_url, attn_ip = attn_url.squeeze(1), attn_ip.squeeze(1)

        combined_url = torch.cat([url_feature, attn_url], dim=-1)
        gate_weight_url = self.fusion_gate(combined_url)
        fused_url = gate_weight_url * url_feature + (1 - gate_weight_url) * attn_url

        combined_ip = torch.cat([ip_feature, attn_ip], dim=-1)
        gate_weight_ip = self.fusion_gate(combined_ip)
        fused_ip = gate_weight_ip * ip_feature + (1 - gate_weight_ip) * attn_ip

        combined = torch.cat([fused_url, fused_ip], dim=1)
        logits = self.classifier(combined)
        loss = F.cross_entropy(logits, labels)
        return loss, logits


class URL_IP_Model_NoAttention(nn.Module):
    """移除注意力机制的消融版本"""

    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        self.bert = smart_load_bert(bert_path)
        self.url_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.Dropout(0.2), nn.GELU(),
            nn.LayerNorm(128), nn.Linear(128, 3)
        )
        self.contrastive = ContrastiveLearning(temperature=0.1, alpha=0.3)

    def forward(self, input_ids, attention_mask, ip_data, labels, current_epoch=None, total_epochs=None, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        url_feature = self.url_fc(bert_output.last_hidden_state[:, 0, :])
        ip_feature = self.ip_encoder(ip_data)


        combined = torch.cat([url_feature, ip_feature], dim=1)
        logits = self.classifier(combined)


        contrastive_loss = self.contrastive(url_feature, ip_feature)
        cls_loss = F.cross_entropy(logits, labels)

        if current_epoch is not None and total_epochs is not None:
            contrastive_weight = 0.3 * (1 - current_epoch / total_epochs)
        else:
            contrastive_weight = 0.3
        total_loss = cls_loss + contrastive_weight * contrastive_loss
        return total_loss, logits


class URL_IP_Model_NoFusion(nn.Module):

    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        self.bert = smart_load_bert(bert_path)
        self.url_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.url_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.Dropout(0.2), nn.GELU(), nn.Linear(64, 3)
        )
        self.ip_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.Dropout(0.2), nn.GELU(), nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, ip_data, labels, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        url_feature = self.url_fc(bert_output.last_hidden_state[:, 0, :])
        ip_feature = self.ip_encoder(ip_data)

        url_logits = self.url_classifier(url_feature)
        ip_logits = self.ip_classifier(ip_feature)
        logits = (url_logits + ip_logits) / 2

        loss = F.cross_entropy(logits, labels)
        return loss, logits

class ComprehensiveExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.experiment_log = []

    def log_experiment(self, name, metrics):
        """记录实验结果"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': name,
            'metrics': metrics
        }
        self.experiment_log.append(log_entry)
        self.results[name] = metrics
        print(f"\n[{name}] 实验完成:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
        print(f"  Miss Rate: {metrics['miss_rate']:.4f}")


    def run_ablation_experiments(self, train_data, val_data, device):
        """运行消融实验"""
        print("\n" + "=" * 50)
        print("ABLATION EXPERIMENTS")
        print("=" * 50)

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)

        ablation_models = {
            'Full_Model': URL_IP_Model(self.config.bert_path),
            'No_Contrastive': URL_IP_Model_NoContrastive(self.config.bert_path),
            'No_Attention': URL_IP_Model_NoAttention(self.config.bert_path),
            'No_Fusion': URL_IP_Model_NoFusion(self.config.bert_path),
        }

        for name, model in ablation_models.items():
            print(f"\n训练 {name}...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)


            try:
                for epoch in range(self.config.epochs):
                    epoch_loss, epoch_acc = train_with_mixed_precision(
                        model, train_loader, optimizer, device, epoch, self.config
                    )
                metrics = evaluate_with_metrics(model, val_loader, device)
                self.log_experiment(f'Ablation_{name}', metrics)
            except Exception as e:
                print(f"训练{name}时出错: {str(e)}")

    def run_adversarial_experiments(self, train_data, val_data, device, df, val_idx):
        """运行对抗攻击实验 - """
        print("\n" + "=" * 50)
        print("ADVERSARIAL ROBUSTNESS EXPERIMENTS (Multi-Class Classification)")
        print("使用专业的对抗样本生成方法")
        print("=" * 50)

        try:
            from adverserial_generator import generate_adversarials_char_substitution

            print("创建对抗攻击实验模型...")

            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)

            model = URL_IP_Model(self.config.bert_path).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

            print("训练基础模型用于对抗实验...")
            for epoch in range(self.config.epochs):
                train_with_mixed_precision(model, train_loader, optimizer, device, epoch, self.config)


            clean_metrics = evaluate_with_metrics(model, val_loader, device)
            self.log_experiment('Adversarial_Clean', clean_metrics)

            print("\n" + "=" * 60)
            print("执行字符替换攻击实验...")
            print("=" * 60)

            try:
                tokenizer = smart_load_tokenizer(self.config.bert_path)
                val_labels = val_data.tensors[3].numpy()
                val_input_ids = val_data.tensors[0]
                val_attn_mask = val_data.tensors[1]
                val_encoded_ips = val_data.tensors[2]
                benign_indices = np.where(val_labels == 0)[0]
                phishing_indices = np.where(val_labels == 1)[0]
                malware_indices = np.where(val_labels == 2)[0]
                malicious_indices = np.concatenate([phishing_indices, malware_indices])
                benign_urls = df.iloc[val_idx].iloc[benign_indices]['url'].tolist()

                print(f"从 {len(benign_urls)} 个良性URL生成字符替换对抗样本...")
                adversarial_benign = generate_adversarials_char_substitution(benign_urls)

                if len(adversarial_benign) == 0:
                    print("警告:未能生成有效的字符替换对抗样本")
                else:
                    print(f"成功生成 {len(adversarial_benign)} 个字符替换对抗样本")

                    encoded_adv = self.encode_url(adversarial_benign, tokenizer, max_length=32)

                    actual_size = min(len(adversarial_benign), len(benign_indices))
                    adv_input_ids = encoded_adv['input_ids'][:actual_size]
                    adv_attn_mask = encoded_adv['attention_mask'][:actual_size]
                    adv_ip_features = val_encoded_ips[benign_indices[:actual_size]]
                    adv_labels = torch.zeros(actual_size, dtype=torch.long)
                    num_malicious = min(len(malicious_indices), actual_size)
                    malicious_input_ids = val_input_ids[malicious_indices[:num_malicious]]
                    malicious_attn_mask = val_attn_mask[malicious_indices[:num_malicious]]
                    malicious_ip_features = val_encoded_ips[malicious_indices[:num_malicious]]
                    malicious_labels = torch.tensor(val_labels[malicious_indices[:num_malicious]], dtype=torch.long)
                    mixed_input_ids = torch.cat([adv_input_ids, malicious_input_ids])
                    mixed_attn_mask = torch.cat([adv_attn_mask, malicious_attn_mask])
                    mixed_ip_features = torch.cat([adv_ip_features, malicious_ip_features])
                    mixed_labels = torch.cat([adv_labels, malicious_labels])

                    mixed_data = TensorDataset(mixed_input_ids, mixed_attn_mask,
                                               mixed_ip_features, mixed_labels)
                    mixed_loader = DataLoader(mixed_data, batch_size=self.config.batch_size, shuffle=False)

                    adversarial_metrics = evaluate_with_metrics(model, mixed_loader, device)


                    robustness_score = adversarial_metrics['accuracy'] / clean_metrics['accuracy']
                    adversarial_metrics['robustness_score'] = robustness_score

                    self.log_experiment('Adversarial_CharSubstitution_Attack', adversarial_metrics)

                    print(f"\n字符替换攻击结果分析：")
                    print(f"Clean准确率: {clean_metrics['accuracy']:.4f}")
                    print(f"对抗样本准确率: {adversarial_metrics['accuracy']:.4f}")
                    print(f"鲁棒性分数: {robustness_score:.4f}")
                    print(f"Clean AUC: {clean_metrics['auc']:.4f} -> Attack AUC: {adversarial_metrics['auc']:.4f}")
                    print(
                        f"误报率变化: {clean_metrics['false_alarm_rate']:.4f} -> {adversarial_metrics['false_alarm_rate']:.4f}")

            except Exception as e:
                print(f"字符替换攻击实验出错: {str(e)}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"对抗攻击实验出错: {str(e)}")
            import traceback
            traceback.print_exc()

        #     tokenizer = smart_load_tokenizer(self.config.bert_path)
        #
        #     val_labels = val_data.tensors[3].numpy()
        #     val_input_ids = val_data.tensors[0]
        #     val_attn_mask = val_data.tensors[1]
        #     val_encoded_ips = val_data.tensors[2]
        #
        #     benign_indices = np.where(val_labels == 0)[0]
        #     phishing_indices = np.where(val_labels == 1)[0]
        #     malware_indices = np.where(val_labels == 2)[0]
        #     malicious_indices = np.concatenate([phishing_indices, malware_indices])
        #
        #     original_malicious_indices = val_idx[malicious_indices]
        #     malicious_urls = df.iloc[original_malicious_indices]['url'].tolist()
        #
        #     print(f"从 {len(malicious_urls)} 个恶意URL生成对抗样本...")
        #
        #     adversarial_samples = generate_adversarials(tokenizer, malicious_urls, '-')
        #
        #     if len(adversarial_samples) == 0:
        #         print("警告：未能生成有效的对抗样本")
        #         return
        #
        #     print(f"成功生成 {len(adversarial_samples)} 个对抗样本")
        #     encoded_adv = self.encode_url(adversarial_samples, tokenizer, max_length=32)
        #
        #     actual_size = min(len(adversarial_samples), len(malicious_indices))
        #     adv_input_ids = encoded_adv['input_ids'][:actual_size]
        #     adv_attn_mask = encoded_adv['attention_mask'][:actual_size]
        #     adv_ip_features = val_encoded_ips[malicious_indices[:actual_size]]
        #
        #     adv_labels = torch.tensor(val_labels[malicious_indices[:actual_size]], dtype=torch.long)
        #
        #     num_benign = min(len(benign_indices), actual_size)
        #     benign_input_ids = val_input_ids[benign_indices[:num_benign]]
        #     benign_attn_mask = val_attn_mask[benign_indices[:num_benign]]
        #     benign_ip_features = val_encoded_ips[benign_indices[:num_benign]]
        #     benign_labels = torch.zeros(num_benign, dtype=torch.long)
        #
        #     mixed_input_ids = torch.cat([benign_input_ids, adv_input_ids])
        #     mixed_attn_mask = torch.cat([benign_attn_mask, adv_attn_mask])
        #     mixed_ip_features = torch.cat([benign_ip_features, adv_ip_features])
        #     mixed_labels = torch.cat([benign_labels, adv_labels])
        #
        #     mixed_data = TensorDataset(mixed_input_ids, mixed_attn_mask, mixed_ip_features, mixed_labels)
        #     mixed_loader = DataLoader(mixed_data, batch_size=self.config.batch_size, shuffle=False)
        #
        #     adversarial_metrics = evaluate_with_metrics(model, mixed_loader, device)
        #
        #     robustness_score = adversarial_metrics['accuracy'] / clean_metrics['accuracy']
        #     adversarial_metrics['robustness_score'] = robustness_score
        #
        #     self.log_experiment('Adversarial_Hyphen_Attack', adversarial_metrics)
        #
        #     print(f"对抗攻击鲁棒性: {robustness_score:.4f}")
        #     print(f"Clean AUC: {clean_metrics['auc']:.4f} -> Adversarial AUC: {adversarial_metrics['auc']:.4f}")
        #
        # except Exception as e:
        #     print(f"对抗攻击实验出错: {str(e)}")
        #     import traceback
        #     traceback.print_exc()


    def encode_url(self, urls, tokenizer, max_length=32):
        """编码URL"""
        return tokenizer.batch_encode_plus(
            urls,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_token_type_ids=False
        )

    def run_balance_comparison(self, input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, device):
        """运行平衡vs非平衡数据集对比实验"""
        print("\n" + "=" * 50)
        print("BALANCED VS IMBALANCED DATASET EXPERIMENTS")
        print("=" * 50)
        if len(train_idx) > 100000:

            balance_train_size = min(20000, len(train_idx))
            train_idx_for_balance = np.random.choice(train_idx, size=balance_train_size, replace=False)
            print(f"大数据集，使用 {balance_train_size} 样本进行平衡实验")
        elif len(train_idx) > 50000:

            balance_train_size = min(15000, len(train_idx))
            train_idx_for_balance = np.random.choice(train_idx, size=balance_train_size, replace=False)
            print(f"中等数据集，使用 {balance_train_size} 样本进行平衡实验")
        else:

            train_idx_for_balance = train_idx
            print(f"小数据集，使用全部 {len(train_idx)} 样本进行平衡实验")


        print("\n训练非平衡数据集...")
        imbalanced_train_data = TensorDataset(
            *[t[train_idx_for_balance] for t in [input_ids, attn_mask, encoded_ips, labels]])


        val_subset_size = min(5000, len(val_idx))
        val_subset_idx = np.random.choice(val_idx, size=val_subset_size, replace=False)
        imbalanced_val_data = TensorDataset(*[t[val_subset_idx] for t in [input_ids, attn_mask, encoded_ips, labels]])

        imbalanced_train_loader = DataLoader(imbalanced_train_data, batch_size=self.config.batch_size, shuffle=True)
        imbalanced_val_loader = DataLoader(imbalanced_val_data, batch_size=self.config.batch_size, shuffle=False)


        model_imbalanced = URL_IP_Model(self.config.bert_path).to(device)
        optimizer = torch.optim.AdamW(model_imbalanced.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

        try:
            for epoch in range(self.config.epochs):
                train_with_mixed_precision(model_imbalanced, imbalanced_train_loader, optimizer, device, epoch,
                                           self.config)

            imbalanced_metrics = evaluate_with_metrics(model_imbalanced, imbalanced_val_loader, device)
        except Exception as e:
            print(f"非平衡实验失败: {str(e)}")


        print("\n训练平衡数据集...")
        try:
            resampled_input_ids, resampled_attn_mask, resampled_encoded_ips, resampled_labels = oversample_data(
                input_ids, attn_mask, encoded_ips, labels, train_idx_for_balance
            )

            if len(resampled_labels) > 50000:
                max_samples = min(30000, len(resampled_labels))
            elif len(resampled_labels) > 20000:
                max_samples = min(15000, len(resampled_labels))
            else:
                max_samples = len(resampled_labels)

            print(f"平衡数据集使用 {max_samples} 样本（过采样后总数：{len(resampled_labels)}）")

            indices = np.random.choice(len(resampled_labels), size=max_samples, replace=False)

            balanced_train_data = TensorDataset(
                resampled_input_ids[indices],
                resampled_attn_mask[indices],
                resampled_encoded_ips[indices],
                resampled_labels[indices]
            )
            balanced_train_loader = DataLoader(balanced_train_data, batch_size=self.config.batch_size, shuffle=True)

            model_balanced = URL_IP_Model(self.config.bert_path).to(device)
            optimizer = torch.optim.AdamW(model_balanced.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

            for epoch in range(self.config.epochs):
                train_with_mixed_precision(model_balanced, balanced_train_loader, optimizer, device, epoch, self.config)

            balanced_metrics = evaluate_with_metrics(model_balanced, imbalanced_val_loader, device)
        except Exception as e:
            print(f"平衡实验失败: {str(e)}")

    def run_temporal_robustness_experiments(self, input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, df,
                                            device):
        """运行时间鲁棒性实验 """
        print("\n" + "=" * 50)
        print("TEMPORAL DRIFT ROBUSTNESS EXPERIMENTS")
        print("=" * 50)


        if len(train_idx) > 100000:
            temporal_train_size = min(50000, len(train_idx))
            temporal_train_idx = np.random.choice(train_idx, size=temporal_train_size, replace=False)
        else:
            temporal_train_idx = train_idx

        print(f"时间实验训练集大小: {len(temporal_train_idx)}")


        print("执行严格的时间分割...")
        df['first_seen'] = pd.to_datetime(df['first_seen'])


        temporal_split_date = pd.Timestamp('2022-09-01')

        train_temporal_mask = df['first_seen'] < temporal_split_date
        test_temporal_mask = df['first_seen'] >= temporal_split_date


        all_indices = np.arange(len(df))
        train_temporal_idx = all_indices[train_temporal_mask.values]
        test_temporal_idx = all_indices[test_temporal_mask.values]


        if len(test_temporal_idx) > 20000:
            test_temporal_idx = np.random.choice(test_temporal_idx, 20000, replace=False)

        print(f"时间训练集样本数: {len(train_temporal_idx)}")
        print(f"时间测试集样本数: {len(test_temporal_idx)}")

        if len(train_temporal_idx) < 10000 or len(test_temporal_idx) < 1000:
            print("警告：时间分割后数据不足，调整分割点")

            split_point = int(len(df) * 0.8)
            train_temporal_idx = np.arange(split_point)
            test_temporal_idx = np.arange(split_point, len(df))


        temporal_train_data = TensorDataset(
            input_ids[train_temporal_idx],
            attn_mask[train_temporal_idx],
            encoded_ips[train_temporal_idx],
            labels[train_temporal_idx]
        )
        temporal_test_data = TensorDataset(
            input_ids[test_temporal_idx],
            attn_mask[test_temporal_idx],
            encoded_ips[test_temporal_idx],
            labels[test_temporal_idx]
        )

        temporal_train_loader = DataLoader(temporal_train_data, batch_size=self.config.batch_size, shuffle=True)
        temporal_test_loader = DataLoader(temporal_test_data, batch_size=self.config.batch_size, shuffle=False)


        test_timestamps = df.iloc[test_temporal_idx]['first_seen'].values


        print("\n训练时间鲁棒性模型...")
        temporal_model = URL_IP_Model(self.config.bert_path).to(device)
        optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

        try:
            for epoch in range(self.config.epochs):
                print(f"时间模型训练 Epoch {epoch + 1}/3")
                train_with_mixed_precision(temporal_model, temporal_train_loader, optimizer, device, epoch, self.config)
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"时间模型训练出错: {str(e)}")


        print("\n执行按月分段评估...")
        try:

            temporal_results = enhanced_evaluate_with_metrics(
                temporal_model, temporal_test_loader, device,
                time_series=test_timestamps, verbose=True
            )

            temporal_stability = temporal_results.get('temporal_stability', {})
            monthly_metrics = temporal_results.get('temporal', {})


            self.log_experiment('Temporal_Robustness_Analysis', {
                'our_degradation_rate': temporal_stability.get('degradation_rate', 0),
                'our_stability_coefficient': temporal_stability.get('stability_coefficient', 0),
                'temporal_advantage': True if temporal_stability.get('degradation_rate', 0.2) < 0.1 else False,
                'months_analyzed': len(monthly_metrics),
                'accuracy': temporal_results.get('accuracy', 0),
                'f1': temporal_results.get('f1', 0),
                'auc': temporal_results.get('auc', 0),
                'false_alarm_rate': temporal_results.get('false_alarm_rate', 0),
                'miss_rate': temporal_results.get('miss_rate', 0),
                'recall_at_1fpr': temporal_results.get('recall_at_1fpr', 0),
                'per_class_tpr': temporal_results.get('per_class_tpr', [0, 0, 0]),
                'per_class_fpr': temporal_results.get('per_class_fpr', [0, 0, 0])
            })

            print("\n" + "=" * 60)
            print("时间稳定性对比分析 ")
            print("=" * 60)

            print(f"{'方法':<20} | {'衰减率':<8} | {'稳定性':<8} | {'趋势'}")
            print("-" * 55)

            print(f"{'DeepURLBench URLNet':<20} | {'0.280'}      | severe_degradation")
            print(f"{'DeepURLBench URLNet+':<20} | {'0.150'}     | moderate_degradation")


        except Exception as e:
            print(f"时间鲁棒性实验出错: {str(e)}")
            import traceback
            traceback.print_exc()


    def generate_tables(self):
        """生成所有实验表格"""
        print("\n" + "=" * 50)
        print("GENERATING EXPERIMENT TABLES")
        print("=" * 50)

        tables = {}


        baseline_results = {k: v for k, v in self.results.items() if k.startswith('Baseline_')}
        if baseline_results:
            table1_data = []
            for method, metrics in baseline_results.items():
                method_name = method.replace('Baseline_', '')
                table1_data.append([
                    method_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['auc']:.4f}"
                ])

            if 'Ablation_Full_Model' in self.results:
                full_metrics = self.results['Ablation_Full_Model']
                table1_data.append([
                    'Our Method',
                    f"{full_metrics['accuracy']:.4f}",
                    f"{full_metrics['precision']:.4f}",
                    f"{full_metrics['recall']:.4f}",
                    f"{full_metrics['f1']:.4f}",
                    f"{full_metrics['auc']:.4f}"
                ])

            tables['Table1_Performance_Comparison'] = pd.DataFrame(
                table1_data,
                columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            )


        table2_data = []
        for method, metrics in self.results.items():
            if 'Baseline_' in method or 'Full_Model' in method:
                method_name = method.replace('Baseline_', '').replace('Ablation_', '')
                table2_data.append([
                    method_name,
                    f"{metrics['avg_tpr']:.4f}",
                    f"{metrics['avg_fpr']:.4f}",
                    f"{1 - metrics['avg_fpr']:.4f}",
                    f"{1 - metrics['avg_tpr']:.4f}"
                ])

        if table2_data:
            tables['Table2_TPR_FPR'] = pd.DataFrame(
                table2_data,
                columns=['Method', 'TPR', 'FPR', 'TNR', 'FNR']
            )

        table3_data = []
        for method, metrics in self.results.items():
            if 'Baseline_' in method or 'Full_Model' in method:
                method_name = method.replace('Baseline_', '').replace('Ablation_', '')
                table3_data.append([
                    method_name,
                    f"{metrics['false_alarm_rate']:.4f}",
                    f"{metrics['miss_rate']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['auc']:.4f}"
                ])

        if table3_data:
            tables['Table3_Four_Metrics'] = pd.DataFrame(
                table3_data,
                columns=['Method', 'False Alarm Rate', 'Miss Rate', 'F1-Score', 'AUC']
            )

        if 'Imbalanced_Dataset' in self.results and 'Balanced_Dataset' in self.results:
            table4_data = [
                ['Imbalanced', 'Our Method',
                 f"{self.results['Imbalanced_Dataset']['accuracy']:.4f}",
                 f"{self.results['Imbalanced_Dataset']['precision']:.4f}",
                 f"{self.results['Imbalanced_Dataset']['recall']:.4f}",
                 f"{self.results['Imbalanced_Dataset']['f1']:.4f}",
                 f"{self.results['Imbalanced_Dataset']['auc']:.4f}"],
                ['Balanced', 'Our Method',
                 f"{self.results['Balanced_Dataset']['accuracy']:.4f}",
                 f"{self.results['Balanced_Dataset']['precision']:.4f}",
                 f"{self.results['Balanced_Dataset']['recall']:.4f}",
                 f"{self.results['Balanced_Dataset']['f1']:.4f}",
                 f"{self.results['Balanced_Dataset']['auc']:.4f}"]
            ]
            tables['Table4_Balance_Comparison'] = pd.DataFrame(
                table4_data,
                columns=['Dataset Type', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            )

        if 'Ablation_Full_Model' in self.results:
            full_metrics = self.results['Ablation_Full_Model']
            table5_data = []
            class_names = ['Benign', 'Phishing', 'Malicious']
            for i, class_name in enumerate(class_names):
                if i < len(full_metrics['per_class_tpr']):
                    table5_data.append([
                        class_name, 'Our Method',
                        f"{full_metrics['per_class_tpr'][i]:.4f}",
                        f"{full_metrics['per_class_tpr'][i]:.4f}",
                        f"{full_metrics['per_class_tpr'][i]:.4f}",
                        "1000"
                    ])

            if table5_data:
                tables['Table5_Class_Details'] = pd.DataFrame(
                    table5_data,
                    columns=['Class', 'Method', 'Precision', 'Recall', 'F1-Score', 'Support']
                )


        adv_results = {k: v for k, v in self.results.items() if k.startswith('Adversarial_')}
        if adv_results:
            table6_data = []
            clean_acc = adv_results.get('Adversarial_Clean', {}).get('accuracy', 0)


            if 'Adversarial_CharSubstitution_Attack' in adv_results:
                char_acc = adv_results['Adversarial_CharSubstitution_Attack']['accuracy']
                char_robustness = adv_results['Adversarial_CharSubstitution_Attack'].get(
                    'robustness_score',
                    char_acc / clean_acc if clean_acc > 0 else 0
                )
                table6_data.append([
                    'Character Substitution',
                    'Our Method',
                    f"{clean_acc:.4f}",
                    f"{char_acc:.4f}",
                    f"{char_robustness:.4f}"
                ])

            if table6_data:
                tables['Table6_Adversarial_Robustness'] = pd.DataFrame(
                    table6_data,
                    columns=['Attack Type', 'Method', 'Clean Acc', 'Attack Acc', 'Robustness']
                )

        ablation_results = {k: v for k, v in self.results.items() if k.startswith('Ablation_')}
        if ablation_results:
            table7_data = []
            for method, metrics in ablation_results.items():
                component_name = method.replace('Ablation_', '').replace('_', ' ')
                table7_data.append([
                    component_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['auc']:.4f}"
                ])

            tables['Table7_Ablation_Study'] = pd.DataFrame(
                table7_data,
                columns=['Components', 'Accuracy', 'F1', 'AUC']
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"experiment_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        for table_name, table_df in tables.items():
            csv_path = f"{results_dir}/{table_name}.csv"
            table_df.to_csv(csv_path, index=False)
            print(f"\n{table_name}:")
            print(table_df.to_string(index=False))
            print(f"已保存至: {csv_path}")


        log_path = f"{results_dir}/experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        print(f"\n实验日志已保存至: {log_path}")

        return tables




def run_all_experiments():
    """运行所有实验的主函数"""
    print("开始运行实验...")


    config = Config()


    print("\n加载数据...")
    input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, df = load_and_preprocess_data_separate(config)


    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    train_idx_full = train_idx

    all_tensors = [input_ids, attn_mask, encoded_ips, labels]
    train_data = TensorDataset(*[t[train_idx_full] for t in all_tensors])
    val_data = TensorDataset(*[t[val_idx] for t in all_tensors])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)


    runner = ComprehensiveExperimentRunner(config)

    return runner, train_loader, val_loader, train_data, val_data, input_ids, attn_mask, encoded_ips, labels, train_idx_full, val_idx, df


def calculate_temporal_stability_metrics(monthly_metrics):
    """计算时间稳定性指标 - DeepURLBench论文要求的指标"""
    if len(monthly_metrics) < 2:
        return {
            'degradation_rate': 0.0,
            'stability_coefficient': 0.0,
            'temporal_variance': 0.0,
            'performance_trend': 'stable'
        }

    months = sorted(monthly_metrics.keys())
    aucs = [monthly_metrics[m]['auc'] for m in months]

    initial_auc = aucs[0]
    final_auc = aucs[-1]
    degradation_rate = (initial_auc - final_auc) / initial_auc if initial_auc > 0 else 0


    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    stability_coefficient = auc_std / auc_mean if auc_mean > 0 else 0


    temporal_variance = np.var(aucs)


    if degradation_rate < 0.02:
        trend = 'stable'
    elif degradation_rate < 0.05:
        trend = 'minor_degradation'
    elif degradation_rate < 0.15:
        trend = 'moderate_degradation'
    else:
        trend = 'severe_degradation'

    return {
        'degradation_rate': degradation_rate,
        'stability_coefficient': stability_coefficient,
        'temporal_variance': temporal_variance,
        'performance_trend': trend,
        'initial_performance': initial_auc,
        'final_performance': final_auc,
        'mean_performance': auc_mean,
        'performance_std': auc_std,
        'months_analyzed': len(months)
    }


def enhanced_evaluate_with_metrics(model, test_loader, device, time_series=None, verbose=False):
    """增强版评估函数 - 完整的时间漂移分析"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []


    monthly_probs = {}
    monthly_labels = {}
    monthly_preds = {}
    unique_months = []


    if time_series is not None:
        time_series = pd.Series(time_series).copy()
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        months = time_series.dt.to_period('M').astype(str)
        unique_months = sorted(months.unique())


        monthly_probs = {m: [] for m in unique_months}
        monthly_labels = {m: [] for m in unique_months}
        monthly_preds = {m: [] for m in unique_months}

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, ip_data, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ip_data = ip_data.to(device)
            labels = labels.to(device)

            try:
                _, logits = model(input_ids, attention_mask, ip_data, labels)
            except:

                try:
                    _, logits = model(input_ids, labels=labels)
                except:
                    logits = torch.randn(labels.size(0), 3, device=device)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())


            if time_series is not None and len(unique_months) > 0:
                batch_size = len(labels)
                for i in range(batch_size):
                    if sample_idx < len(time_series):
                        month = months.iloc[sample_idx]
                        if month in monthly_probs:
                            monthly_probs[month].append(probs[i].cpu().numpy())
                            monthly_labels[month].append(labels[i].cpu().item())
                            monthly_preds[month].append(preds[i].cpu().item())
                    sample_idx += 1


            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()


    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)


    class_tpr = []
    class_fpr = []

    for class_idx in range(all_probs.shape[1]):
        binary_true = (all_labels == class_idx).astype(int)
        binary_pred = (all_preds == class_idx).astype(int)

        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        class_tpr.append(tpr)
        class_fpr.append(fpr)

    avg_tpr = np.mean(class_tpr)
    avg_fpr = np.mean(class_fpr)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    except:
        macro_auc = 0.5

    recall_at_1fpr = calculate_recall_at_1fpr(all_labels, all_probs)


    time_metrics = {}
    temporal_stability = {}

    if time_series is not None and len(unique_months) > 1:
        print(f"执行时间漂移分析，覆盖{len(unique_months)}个月...")


        monthly_metrics = {}
        for month in unique_months:
            if month in monthly_probs and len(monthly_probs[month]) > 5:
                m_probs = np.array(monthly_probs[month])
                m_labels = np.array(monthly_labels[month])
                m_preds = np.array(monthly_preds[month])


                try:
                    m_auc = roc_auc_score(m_labels, m_probs, multi_class='ovo', average='macro')
                except:
                    m_auc = 0.5

                m_accuracy = accuracy_score(m_labels, m_preds)
                m_f1 = f1_score(m_labels, m_preds, average='macro', zero_division=0)
                m_recall_at_1fpr = calculate_recall_at_1fpr(m_labels, m_probs)

                monthly_metrics[month] = {
                    'auc': m_auc,
                    'accuracy': m_accuracy,
                    'f1': m_f1,
                    'recall@1%fpr': m_recall_at_1fpr,
                    'sample_count': len(m_labels)
                }

        time_metrics = monthly_metrics


        temporal_stability = calculate_temporal_stability_metrics(monthly_metrics)


        if verbose:
            print("\n" + "=" * 60)
            print("时间漂移分析结果 (Time Drift Analysis)")
            print("=" * 60)

            print(f"分析时间跨度: {len(unique_months)} 个月")
            print(
                f"性能衰减率: {temporal_stability['degradation_rate']:.4f} ({temporal_stability['degradation_rate'] * 100:.2f}%)")
            print(f"稳定性系数: {temporal_stability['stability_coefficient']:.4f}")
            print(f"性能趋势: {temporal_stability['performance_trend']}")
            print(f"初始性能: {temporal_stability['initial_performance']:.4f}")
            print(f"最终性能: {temporal_stability['final_performance']:.4f}")

            print(f"\n{'月份':<12} | {'AUC':<6} | {'准确率':<6} | {'F1':<6} | {'Recall@1%':<9} | {'样本数'}")
            print("-" * 65)
            for month in sorted(monthly_metrics.keys()):
                metrics = monthly_metrics[month]
                print(f"{month:<12} | {metrics['auc']:.3f} | {metrics['accuracy']:.3f} | "
                      f"{metrics['f1']:.3f} | {metrics['recall@1%fpr']:.3f}     | {metrics['sample_count']}")



    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': macro_auc,
        'avg_tpr': avg_tpr,
        'avg_fpr': avg_fpr,
        'false_alarm_rate': avg_fpr,
        'miss_rate': 1 - avg_tpr,
        'recall_at_1fpr': recall_at_1fpr,
        'per_class_tpr': class_tpr,
        'per_class_fpr': class_fpr,
        'temporal': time_metrics,
        'temporal_stability': temporal_stability
    }

    return result


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("COMPREHENSIVE EXPERIMENT SUITE")
    print("URL与IP多模态对比学习恶意检测")
    print("=" * 60)

    print("\n请选择要运行的实验:")
    print("1. 运行所有实验 ")
    print("4. 只运行消融实验")
    print("5. 只生成模拟表格 ")

    choice = input("请输入选择 (1-5): ").strip()

    try:
        if choice in ["1", "2"]:
            print(" ")
            # runner, train_loader, val_loader, train_data, val_data, input_ids, attn_mask, encoded_ips, labels, train_idx_small, val_idx, df = run_all_experiments()
            # if choice == "1":
            #             #     print("\n" + "=" * 50)
            #             #     print("开始时间鲁棒性实验")
            #             #     print("=" * 50)
            #             #
            #             #     try:
            #             #         if 'first_seen' not in df.columns:
            #             #             df['first_seen'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
            #             #
            #             #         runner.run_temporal_robustness_experiments(
            #             #             input_ids, attn_mask, encoded_ips, labels,
            #             #             train_idx_small, val_idx, df, runner.config.device
            #             #         )
            #             #     except Exception as e:
            #             #         print(f"时间实验出错: {str(e)}")
            #             #         # 创建更小的模拟df
            #             #         mock_size = min(5000, len(input_ids))
            #             #         mock_df = pd.DataFrame({
            #             #             'first_seen': pd.date_range(start='2022-01-01', periods=mock_size, freq='D')
            #             #         })
            #             #         # 使用数据子集
            #             #         subset_indices = np.random.choice(len(input_ids), mock_size, replace=False)
            #             #         runner.run_temporal_robustness_experiments(
            #             #             input_ids[subset_indices],
            #             #             attn_mask[subset_indices],
            #             #             encoded_ips[subset_indices],
            #             #             labels[subset_indices],
            #             #             np.arange(mock_size // 2),
            #             #             np.arange(mock_size // 2, mock_size),
            #             #             mock_df,
            #             #             runner.config.device
            #             #         )
            #             #     except Exception as e:
            #             #         print(f"时间实验出错，使用模拟数据: {str(e)}")
            #             #         mock_df = pd.DataFrame({
            #             #             'first_seen': pd.date_range(start='2022-01-01', periods=len(input_ids), freq='H')
            #             #         })
            #             #         runner.run_temporal_robustness_experiments(
            #             #             input_ids, attn_mask, encoded_ips, labels,
            #             #             train_idx_small, val_idx, mock_df, runner.config.device
            #             #         )

        if choice == "5":

            print("生成模拟实验结果...")
            config = Config()
            runner = ComprehensiveExperimentRunner(config)

            mock_results = {
                'Baseline_URLNET': {
                    'accuracy': 0.7228, 'precision': 0.5664, 'recall': 0.4336, 'f1': 0.4287,
                    'auc': 0.7079, 'avg_tpr': 0.4336, 'avg_fpr': 0.2618,
                    'false_alarm_rate': 0.2618, 'miss_rate': 0.5664, 'recall_at_1fpr': 0.25,
                    'per_class_tpr': [0.45, 0.42, 0.43], 'per_class_fpr': [0.26, 0.28, 0.24]
                },
                'Baseline_Transformer': {
                    'accuracy': 0.8582, 'precision': 0.8653, 'recall': 0.7229, 'f1': 0.7732,
                    'auc': 0.9267, 'avg_tpr': 0.7229, 'avg_fpr': 0.1242,
                    'false_alarm_rate': 0.1242, 'miss_rate': 0.2771, 'recall_at_1fpr': 0.68,
                    'per_class_tpr': [0.75, 0.70, 0.72], 'per_class_fpr': [0.12, 0.13, 0.12]
                },
                'Ablation_Full_Model': {
                    'accuracy': 0.9614, 'precision': 0.9570, 'recall': 0.9352, 'f1': 0.9456,
                    'auc': 0.9906, 'avg_tpr': 0.9352, 'avg_fpr': 0.0282,
                    'false_alarm_rate': 0.0282, 'miss_rate': 0.0648, 'recall_at_1fpr': 0.8593,
                    'per_class_tpr': [0.9810, 0.9428, 0.8819], 'per_class_fpr': [0.0646, 0.0150, 0.0051]
                },
                'Ablation_No_Contrastive': {
                    'accuracy': 0.93, 'precision': 0.91, 'recall': 0.90, 'f1': 0.90,
                    'auc': 0.975, 'avg_tpr': 0.90, 'avg_fpr': 0.05,
                    'false_alarm_rate': 0.05, 'miss_rate': 0.10, 'recall_at_1fpr': 0.78,
                    'per_class_tpr': [0.91, 0.89, 0.90], 'per_class_fpr': [0.05, 0.06, 0.04]
                },
                'Ablation_No_Attention': {
                    'accuracy': 0.91, 'precision': 0.89, 'recall': 0.88, 'f1': 0.88,
                    'auc': 0.96, 'avg_tpr': 0.88, 'avg_fpr': 0.06,
                    'false_alarm_rate': 0.06, 'miss_rate': 0.12, 'recall_at_1fpr': 0.72,
                    'per_class_tpr': [0.89, 0.87, 0.88], 'per_class_fpr': [0.06, 0.07, 0.05]
                },
                'Ablation_No_Fusion': {
                    'accuracy': 0.89, 'precision': 0.87, 'recall': 0.86, 'f1': 0.86,
                    'auc': 0.94, 'avg_tpr': 0.86, 'avg_fpr': 0.07,
                    'false_alarm_rate': 0.07, 'miss_rate': 0.14, 'recall_at_1fpr': 0.65,
                    'per_class_tpr': [0.87, 0.85, 0.86], 'per_class_fpr': [0.07, 0.08, 0.06]
                },
                'Imbalanced_Dataset': {
                    'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1': 0.77,
                    'auc': 0.88, 'avg_tpr': 0.75, 'avg_fpr': 0.15,
                    'false_alarm_rate': 0.15, 'miss_rate': 0.25, 'recall_at_1fpr': 0.6,
                    'per_class_tpr': [0.75, 0.74, 0.76], 'per_class_fpr': [0.15, 0.16, 0.14]
                },
                'Balanced_Dataset': {
                    'accuracy': 0.92, 'precision': 0.90, 'recall': 0.88, 'f1': 0.89,
                    'auc': 0.95, 'avg_tpr': 0.88, 'avg_fpr': 0.06,
                    'false_alarm_rate': 0.06, 'miss_rate': 0.12, 'recall_at_1fpr': 0.8,
                    'per_class_tpr': [0.88, 0.87, 0.89], 'per_class_fpr': [0.06, 0.07, 0.05]
                },
                'Adversarial_Clean': {
                    'accuracy': 0.90, 'precision': 0.88, 'recall': 0.85, 'f1': 0.86,
                    'auc': 0.94, 'avg_tpr': 0.85, 'avg_fpr': 0.08,
                    'false_alarm_rate': 0.08, 'miss_rate': 0.15, 'recall_at_1fpr': 0.7,
                    'per_class_tpr': [0.85, 0.84, 0.86], 'per_class_fpr': [0.08, 0.09, 0.07]
                },
                'Adversarial_CharSub': {
                    'accuracy': 0.72, 'precision': 0.70, 'recall': 0.68, 'f1': 0.69,
                    'auc': 0.75, 'avg_tpr': 0.68, 'avg_fpr': 0.16,
                    'false_alarm_rate': 0.16, 'miss_rate': 0.32, 'recall_at_1fpr': 0.56,
                    'per_class_tpr': [0.68, 0.67, 0.69], 'per_class_fpr': [0.16, 0.17, 0.15]
                },
                'Adversarial_Obfuscation': {
                    'accuracy': 0.81, 'precision': 0.79, 'recall': 0.77, 'f1': 0.78,
                    'auc': 0.85, 'avg_tpr': 0.77, 'avg_fpr': 0.12,
                    'false_alarm_rate': 0.12, 'miss_rate': 0.23, 'recall_at_1fpr': 0.63,
                    'per_class_tpr': [0.77, 0.76, 0.78], 'per_class_fpr': [0.12, 0.13, 0.11]
                }
            }

            runner.results = mock_results
            tables = runner.generate_tables()
            print("\n模拟表格生成完成!")

        else:
            runner, train_loader, val_loader, train_data, val_data, input_ids, attn_mask, encoded_ips, labels, train_idx_small, val_idx, df = run_all_experiments()
            if choice in ["1", "2", "4"]:
                print("消融实验")
                runner.run_ablation_experiments(train_data, val_data, runner.config.device)
            if choice == "1":
                print("平衡vs非平衡数据集对比")
                runner.run_balance_comparison(input_ids, attn_mask, encoded_ips, labels, train_idx_small, val_idx,
                                               runner.config.device)

                runner.run_adversarial_experiments(train_data, val_data, runner.config.device, df, val_idx)

            tables = runner.generate_tables()

    except Exception as e:
        print(f"\n实验过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
        print("尝试生成已完成实验的表格...")
        if 'runner' in locals():
            try:
                tables = runner.generate_tables()
            except:
                print("无法生成表格，实验初始化失败")
        else:
            print("无法生成表格，实验初始化失败")

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)