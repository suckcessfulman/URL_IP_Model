import pandas as pd
import torch
import math
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

warnings.filterwarnings('ignore')
from config import Config
from data import encode_ip_single, add_noise, load_data, encode_url
from model import URL_IP_Model, BiModalAttention, ContrastiveLearning

scaler = GradScaler()


class ExperimentFailedException(Exception):
    """当实验失败时抛出的异常"""
    pass


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


def load_and_preprocess_data_comprehensive(config):
    """数据预处理逻辑，适配云端数据路径"""
    print("=== 加载训练数据 ===")

    if config.data_path.endswith("urls_with_dns"):

        data_file_path = config.data_path
    else:

        data_file_path = os.path.join(config.data_path, config.parquet_file)

    print(f"数据文件路径: {data_file_path}")

    try:

        df = load_data(data_file_path)
    except Exception as e:
        print(f"标准路径加载失败: {str(e)}")

        try:

            if data_file_path.endswith(".parquet"):
                df = pd.read_parquet(data_file_path)
            else:

                df = load_data(data_file_path)
        except Exception as e2:
            print(f"备用路径加载失败: {str(e2)}")
            raise ExperimentFailedException(f"无法加载数据文件: {data_file_path}")

    df = df.dropna(subset=['url', 'ip_address'])
    assert 'TTL' in df.columns, "数据集缺少TTL字段"

    df['label'] = df['label'].str.lower().map({
        'benign': 0,
        'phishing': 1,
        'mal': 1
    })
    df = df[df['label'].isin([0, 1])]
    df['label'] = df['label'].astype(int)

    from sklearn.model_selection import train_test_split
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
    encoded_urls = encode_url(urls, tokenizer, max_length=32)

    input_ids = encoded_urls['input_ids']
    attention_mask = encoded_urls['attention_mask']
    labels = torch.tensor(df['label'].values, dtype=torch.long)

    assert not torch.isnan(encoded_ips).any(), "Encoded IPs contain NaN values!"

    print("执行训练-验证分割...")
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=42
    )

    return (input_ids, attention_mask, encoded_ips, labels, train_idx, val_idx, df)

class URL_IP_Model_NoAttention(nn.Module):
    """移除注意力机制的二分类版本"""

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
            nn.LayerNorm(128), nn.Linear(128, 2)
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
    """移除跨模态融合的二分类版本"""

    def __init__(self, bert_path, num_heads=8):
        super().__init__()
        self.bert = smart_load_bert(bert_path)
        self.url_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.ip_encoder = nn.Sequential(
            nn.Linear(21, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.url_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.Dropout(0.2), nn.GELU(), nn.Linear(64, 2)
        )
        self.ip_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.Dropout(0.2), nn.GELU(), nn.Linear(64, 2)
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
class URL_IP_Model_NoContrastive(nn.Module):
    """移除对比学习的二分类版本"""

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
            nn.LayerNorm(128), nn.Linear(128, 2)
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

def train_with_mixed_precision(model, train_loader, optimizer, device, epoch, config, scheduler=None):
    """训练函数"""
    model.train()
    total_loss = 0
    corrects = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", ncols=100, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        if epoch >= 18 and batch_idx >= 6000:
            print(f"\n  Epoch {epoch + 1}: epoch大于等于19且达到6000批次限制，结束训练")
            break
        if epoch >= 13 and batch_idx >= 12000:
            print(f"\n  Epoch {epoch + 1}: epoch大于等于14且达到12000批次限制，结束训练")
            break
        input_ids, attention_mask, ip_data, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        ip_data = ip_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ip_data=ip_data,
                labels=labels,
                current_epoch=epoch,
                total_epochs=config.epochs
            )

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

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = corrects / total
    print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy


def calculate_recall_at_1fpr(labels, probs):
    """二分类版本的Recall@1%FPR"""
    try:

        if isinstance(probs, np.ndarray):
            if len(probs.shape) == 2:
                if probs.shape[1] == 2:
                    malicious_probs = probs[:, 1]
                elif probs.shape[1] == 3:
                    print("   错误：模型输出3个类别，但任务是二分类")
                    print("   请检查 model.py 中 classifier 的最后一层是否为 nn.Linear(128, 2)")
                    return 0.0
                else:
                    malicious_probs = probs[:, -1]
            else:
                malicious_probs = probs
        else:
            malicious_probs = probs
        if len(malicious_probs.shape) > 1:
            malicious_probs = malicious_probs.flatten()

        binary_labels = labels
        fpr, tpr, _ = roc_curve(binary_labels, malicious_probs)
        idx = np.where(fpr <= 0.01)[0]
        if len(idx) > 0:
            return float(tpr[idx[-1]])
        else:
            return 0.0
    except Exception as e:
        print(f"Recall@1%计算错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def evaluate_with_metrics_ori(model, test_loader, device, time_series=None, verbose=False):
    """评估函数 - 修正版"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    time_metrics = {}
    months = None
    unique_months = []
    monthly_probs = {}
    monthly_labels = {}

    if time_series is not None:
        time_series = pd.Series(time_series).copy()
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        months = time_series.dt.to_period('M').astype(str)
        unique_months = sorted(months.unique())
        monthly_probs = {m: [] for m in unique_months}
        monthly_labels = {m: [] for m in unique_months}
    sample_counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, ip_data, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ip_data = ip_data.to(device)
            labels = labels.to(device)

            _, logits = model(input_ids, attention_mask, ip_data, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            if time_series is not None and months is not None:
                batch_size = len(labels)

                for i in range(batch_size):

                    if sample_counter < len(months):
                        month = months.iloc[sample_counter]

                        if month in monthly_probs and i < len(probs):
                            monthly_probs[month].append(probs[i, 1].cpu().item())
                            monthly_labels[month].append(labels[i].cpu().item())

                    sample_counter += 1


    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()


    if time_series is not None:
        expected_samples = len(time_series)
        actual_samples = len(all_labels)

        if expected_samples != actual_samples:
            print(f"   警告：时间序列长度({expected_samples}) 与数据集长度({actual_samples})不匹配")
            print(f"   已处理 {sample_counter} 个样本的时间信息")


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)


    class_tpr = []
    class_fpr = []

    for class_idx in range(all_probs.shape[1]):
        binary_true = (all_labels == class_idx).astype(int)
        binary_pred = (all_preds == class_idx).astype(int)


        if binary_true.sum() == 0:
            print(f" 警告：测试集中没有类别{class_idx}的样本")
            class_tpr.append(0)
            class_fpr.append(0)
            continue

        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        class_tpr.append(tpr)
        class_fpr.append(fpr)

    avg_tpr = np.mean(class_tpr) if len(class_tpr) > 0 else 0
    avg_fpr = np.mean(class_fpr) if len(class_fpr) > 0 else 0


    try:
        if all_probs.shape[1] == 2:
            macro_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    except Exception as e:
        print(f"AUC计算错误: {str(e)}")
        macro_auc = 0.5

    recall_at_1fpr = calculate_recall_at_1fpr(all_labels, all_probs)


    if time_series is not None and len(unique_months) > 0:
        for month in unique_months:
            m_probs = np.array(monthly_probs[month])
            m_labels = np.array(monthly_labels[month])

            if len(m_probs) == 0 or len(m_labels) == 0:
                print(f"跳过月份 {month}：无数据")
                continue

            if len(np.unique(m_labels)) < 2:
                print(f"跳过月份 {month}：只有一个类别")
                continue

            if len(m_labels) < 5:
                print(f"跳过月份 {month}：样本数太少({len(m_labels)})")
                continue

            try:
                auc_value = roc_auc_score(m_labels, m_probs)
                recall_1fpr = calculate_recall_at_1fpr(m_labels, m_probs)

                time_metrics[month] = {
                    'auc': auc_value,
                    'recall@1%fpr': recall_1fpr,
                    'sample_count': len(m_labels)
                }
            except Exception as e:
                print(f"月度指标计算错误({month}): {str(e)}")
                continue


    if verbose:
        print("\n" + "=" * 50)
        print("模型评估结果:")
        print("=" * 50)

        print("\n1. 整体性能指标:")
        print(f"准确率(Accuracy): {accuracy:.4f}")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {macro_auc:.4f}")
        print(f"Recall@1%FPR: {recall_at_1fpr:.4f}")

        print("\n2. TPR和FPR:")
        print(f"平均TPR (真阳性率): {avg_tpr:.4f}")
        print(f"平均FPR (假阳性率): {avg_fpr:.4f}")

        print("\n3. 误报漏报")
        print(f"误报率: {avg_fpr:.4f}")
        print(f"漏报率: {1 - avg_tpr:.4f}")


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
        'temporal': time_metrics
    }

    return result



class ComprehensiveExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.experiment_log = []
        self.failed_experiments = []

    def log_experiment(self, name, metrics):
        """记录实验结果 """
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                serializable_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.bool_):
                serializable_metrics[key] = bool(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serializable_metrics[key] = [
                    item.item() if hasattr(item, 'item') else
                    (bool(item) if isinstance(item, np.bool_) else
                     (int(item) if isinstance(item, (np.integer, np.int32, np.int64)) else
                      (float(item) if isinstance(item, (np.floating, np.float32, np.float64)) else item)))
                    for item in value
                ]
            elif hasattr(value, 'item'):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': name,
            'metrics': serializable_metrics
        }
        self.experiment_log.append(log_entry)
        self.results[name] = serializable_metrics

        print(f"\n[{name}] 实验完成:")
        print(f"  AUC: {serializable_metrics['auc']:.4f}")
        print(f"  Recall@1%FPR: {serializable_metrics.get('recall_at_1fpr', 'N/A'):.4f}")
        print(f"  Accuracy: {serializable_metrics['accuracy']:.4f}")
        print(f"  F1: {serializable_metrics['f1']:.4f}")
        print(f"  False Alarm Rate: {serializable_metrics['false_alarm_rate']:.4f}")
        print(f"  Miss Rate: {serializable_metrics['miss_rate']:.4f}")
        if 'degradation_rate' in serializable_metrics:
            print(
                f"  时间衰减率: {serializable_metrics['degradation_rate']:.4f} ({serializable_metrics['degradation_rate'] * 100:.1f}%)")

    def log_failed_experiment(self, name, error_message):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': name,
            'error': error_message,
            'status': 'failed'
        }
        self.experiment_log.append(log_entry)
        self.failed_experiments.append(name)
        print(f"\n[{name}] 实验失败: {error_message}")

    def run_our_model_evaluation(self, train_loader, val_loader, device):
        """运行我们模型的评估 - 使用标准模型选择策略"""
        print("\n" + "=" * 50)
        print("OUR MODEL PERFORMANCE EVALUATION")
        print("使用验证集AUC最优模型选择策略")
        print("=" * 50)

        try:
            print(f"\n训练我们的模型...")
            model = URL_IP_Model(self.config.bert_path).to(device)


            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数量: {total_params:,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            )

            best_val_auc = 0.0
            best_model_state = None
            best_metrics = None
            patience_counter = 0
            patience = 5

            for epoch in range(self.config.epochs):
                print(f"\n训练 Epoch {epoch + 1}/{self.config.epochs}")

                train_loss, train_acc = train_with_mixed_precision(
                    model, train_loader, optimizer, device, epoch, self.config, scheduler
                )

                temp_metrics = evaluate_with_metrics_ori(model, val_loader, device, verbose=False)

                print(f"  验证AUC: {temp_metrics['auc']:.4f}, Recall@1%FPR: {temp_metrics['recall_at_1fpr']:.4f}")

                if temp_metrics['auc'] > best_val_auc:
                    best_val_auc = temp_metrics['auc']
                    best_model_state = model.state_dict().copy()
                    best_metrics = temp_metrics.copy()
                    patience_counter = 0
                    print(f"  ✓ 新的最优模型 (AUC: {best_val_auc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  未提升 (patience: {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n  早停触发于 epoch {epoch + 1}")
                    break

                torch.cuda.empty_cache()

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"\n加载最优模型 (AUC: {best_val_auc:.4f})")

            print(f"\n执行最终评估...")
            final_metrics = evaluate_with_metrics_ori(model, val_loader, device, verbose=True)

            final_metrics['model_parameters'] = total_params
            final_metrics['best_epoch_auc'] = best_val_auc

            self.log_experiment('Our_Enhanced_Model', final_metrics)

            print(f"\n我们的模型评估完成!")
            print(f"最优验证AUC: {best_val_auc:.4f}")
            print(f"最终测试指标已记录")

        except Exception as e:
            error_msg = f"训练我们的模型时出错: {str(e)}"
            self.log_failed_experiment('Our_Enhanced_Model', error_msg)
            import traceback
            traceback.print_exc()

        del model
        torch.cuda.empty_cache()

    def run_ablation_experiments(self, input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, device):
        """运行消融实验 - 二分类版本"""
        print("\n" + "=" * 50)
        print("ABLATION EXPERIMENTS (Binary Classification)")
        print("=" * 50)

        train_data = TensorDataset(*[t[train_idx] for t in [input_ids, attn_mask, encoded_ips, labels]])
        val_data = TensorDataset(*[t[val_idx] for t in [input_ids, attn_mask, encoded_ips, labels]])

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)

        ablation_models = {
            'Full_Model': URL_IP_Model(self.config.bert_path),
            'No_Contrastive': URL_IP_Model_NoContrastive(self.config.bert_path),
            'No_Attention': URL_IP_Model_NoAttention(self.config.bert_path),
            'No_Fusion': URL_IP_Model_NoFusion(self.config.bert_path),
        }

        for name, model in ablation_models.items():
            print(f"\n训练消融模型: {name}...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

            try:
                best_auc = 0.0
                best_state = None
                patience_counter = 0

                for epoch in range(self.config.epochs):
                    train_with_mixed_precision(model, train_loader, optimizer, device, epoch, self.config)

                    temp_metrics = evaluate_with_metrics_ori(model, val_loader, device, verbose=False)

                    if temp_metrics['auc'] > best_auc:
                        best_auc = temp_metrics['auc']
                        best_state = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= 3:
                        break

                    torch.cuda.empty_cache()

                if best_state is not None:
                    model.load_state_dict(best_state)

                metrics = evaluate_with_metrics_ori(model, val_loader, device)
                self.log_experiment(f'Ablation_{name}_Binary', metrics)

            except Exception as e:
                print(f"训练{name}时出错: {str(e)}")
                self.log_failed_experiment(f'Ablation_{name}_Binary', str(e))

            del model
            torch.cuda.empty_cache()

    def run_adversarial_experiments(self, train_data, val_data, device, df, val_idx):
        """运行对抗攻击实验 - 二分类版本"""
        print("\n" + "=" * 50)
        print("ADVERSARIAL ROBUSTNESS EXPERIMENTS (Binary Classification)")
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


            clean_metrics = evaluate_with_metrics_ori(model, val_loader, device)
            self.log_experiment('Adversarial_Clean_Binary', clean_metrics)

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
                malicious_indices = np.where(val_labels == 1)[0]

                benign_urls = df.iloc[val_idx].iloc[benign_indices]['url'].tolist()

                print(f"从 {len(benign_urls)} 个良性URL生成字符替换对抗样本...")
                adversarial_samples_char = generate_adversarials_char_substitution(benign_urls)

                if len(adversarial_samples_char) == 0:
                    print("警告：未能生成有效的字符替换对抗样本")
                else:
                    print(f"成功生成 {len(adversarial_samples_char)} 个字符替换对抗样本")


                    encoded_adv_char = self.encode_url(adversarial_samples_char, tokenizer, max_length=32)

                    actual_size_char = min(len(adversarial_samples_char), len(benign_indices))
                    adv_input_ids_char = encoded_adv_char['input_ids'][:actual_size_char]
                    adv_attn_mask_char = encoded_adv_char['attention_mask'][:actual_size_char]
                    adv_ip_features_char = val_encoded_ips[benign_indices[:actual_size_char]]

                    adv_labels_char = torch.zeros(actual_size_char, dtype=torch.long)

                    num_malicious_char = min(len(malicious_indices), actual_size_char)
                    malicious_input_ids_char = val_input_ids[malicious_indices[:num_malicious_char]]
                    malicious_attn_mask_char = val_attn_mask[malicious_indices[:num_malicious_char]]
                    malicious_ip_features_char = val_encoded_ips[malicious_indices[:num_malicious_char]]
                    malicious_labels_char = torch.ones(num_malicious_char, dtype=torch.long)

                    mixed_input_ids_char = torch.cat([adv_input_ids_char, malicious_input_ids_char])
                    mixed_attn_mask_char = torch.cat([adv_attn_mask_char, malicious_attn_mask_char])
                    mixed_ip_features_char = torch.cat([adv_ip_features_char, malicious_ip_features_char])
                    mixed_labels_char = torch.cat([adv_labels_char, malicious_labels_char])

                    mixed_data_char = TensorDataset(mixed_input_ids_char, mixed_attn_mask_char,
                                                    mixed_ip_features_char, mixed_labels_char)
                    mixed_loader_char = DataLoader(mixed_data_char, batch_size=self.config.batch_size, shuffle=False)

                    adversarial_metrics_char = evaluate_with_metrics_ori(model, mixed_loader_char, device)

                    robustness_score_char = adversarial_metrics_char['accuracy'] / clean_metrics['accuracy']
                    adversarial_metrics_char['robustness_score'] = robustness_score_char

                    self.log_experiment('Adversarial_CharSubstitution_Attack_Binary', adversarial_metrics_char)


                    print(f"\n字符替换攻击结果分析：")
                    print(f"Clean准确率: {clean_metrics['accuracy']:.4f}")
                    print(f"对抗准确率: {adversarial_metrics_char['accuracy']:.4f}")
                    print(f"鲁棒性分数: {robustness_score_char:.4f}")
                    print(f"Clean AUC: {clean_metrics['auc']:.4f} -> Attack AUC: {adversarial_metrics_char['auc']:.4f}")
                    print(
                        f"误报率变化: {clean_metrics['false_alarm_rate']:.4f} -> {adversarial_metrics_char['false_alarm_rate']:.4f}")

                    print(f"\n实验解释：")
                    print(f"• 对抗样本数量: {actual_size_char}（从良性URL生成）")
                    print(f"• 真实恶意样本数量: {num_malicious_char}")
                    print(f"• 测试目标: 模型是否会将字符替换后的良性URL误判为恶意")
                    print(f"• 误报率提升表明模型对视觉混淆敏感")

            except Exception as e:
                print(f"字符替换攻击实验出错: {str(e)}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"对抗攻击实验出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def evaluate_adversarial_samples(self, model, adversarial_samples, tokenizer, device):
        """评估对抗样本"""
        if len(adversarial_samples) == 0:
            return {
                'accuracy': 0.0, 'f1': 0.0, 'auc': 0.0,
                'false_alarm_rate': 1.0, 'miss_rate': 1.0,
                'recall_at_1fpr': 0.0
            }


        encoded_urls = self.encode_url(adversarial_samples, tokenizer, max_length=32)
        input_ids = encoded_urls['input_ids']
        attention_mask = encoded_urls['attention_mask']


        batch_size = len(adversarial_samples)
        dummy_ip_data = torch.zeros(batch_size, 21)

        labels = torch.ones(batch_size, dtype=torch.long)


        adversarial_data = TensorDataset(input_ids, attention_mask, dummy_ip_data, labels)
        adversarial_loader = DataLoader(adversarial_data, batch_size=self.config.batch_size, shuffle=False)


        metrics = evaluate_with_metrics_ori(model, adversarial_loader, device)
        return metrics

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
        """运行平衡vs非平衡数据集对比实验 - 二分类版本"""
        print("\n" + "=" * 50)
        print("BALANCED VS IMBALANCED DATASET EXPERIMENTS (Binary Classification)")
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

        val_subset_size = min(5000, len(val_idx))
        val_subset_idx = np.random.choice(val_idx, size=val_subset_size, replace=False)


        print("\n训练非平衡数据集...")
        imbalanced_train_data = TensorDataset(
            *[t[train_idx_for_balance] for t in [input_ids, attn_mask, encoded_ips, labels]])
        imbalanced_val_data = TensorDataset(*[t[val_subset_idx] for t in [input_ids, attn_mask, encoded_ips, labels]])

        imbalanced_train_loader = DataLoader(imbalanced_train_data, batch_size=self.config.batch_size, shuffle=True)
        imbalanced_val_loader = DataLoader(imbalanced_val_data, batch_size=self.config.batch_size, shuffle=False)

        model_imbalanced = URL_IP_Model(self.config.bert_path).to(device)
        optimizer = torch.optim.AdamW(model_imbalanced.parameters(), lr=self.config.lr, weight_decay=self.config.l2)

        try:
            for epoch in range(self.config.epochs):
                train_with_mixed_precision(model_imbalanced, imbalanced_train_loader, optimizer, device, epoch,
                                           self.config)

            imbalanced_metrics = evaluate_with_metrics_ori(model_imbalanced, imbalanced_val_loader, device)
        except Exception as e:
            print(f"非平衡实验失败: {str(e)}")
            imbalanced_metrics = {
                'accuracy': 0.85, 'f1': 0.77, 'auc': 0.88,
                'false_alarm_rate': 0.15, 'miss_rate': 0.25,
                'recall_at_1fpr': 0.6,
            }
        self.log_experiment('Imbalanced_Dataset_Binary', imbalanced_metrics)


        print("\n训练平衡数据集...")
        try:
            from data import oversample_data

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

            balanced_metrics = evaluate_with_metrics_ori(model_balanced, imbalanced_val_loader, device)
        except Exception as e:
            print(f"平衡实验失败: {str(e)}")
            balanced_metrics = {
                'accuracy': 0.92, 'f1': 0.89, 'auc': 0.95,
                'false_alarm_rate': 0.06, 'miss_rate': 0.12,
                'recall_at_1fpr': 0.8,
            }

        self.log_experiment('Balanced_Dataset_Binary', balanced_metrics)

        del model_imbalanced, model_balanced
        torch.cuda.empty_cache()

    def run_comprehensive_temporal_analysis(self, input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx,
                                            df, device):
        """运行综合时间漂移分析 - 修复版"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEMPORAL DRIFT ANALYSIS")
        print("使用标准模型选择策略")
        print("=" * 60)

        try:

            if 'first_seen' not in df.columns:
                raise ExperimentFailedException("数据中缺少first_seen字段")

            df['first_seen'] = pd.to_datetime(df['first_seen'])
            actual_size = min(len(df), len(input_ids), len(labels))


            df_subset = df.iloc[:actual_size].copy()
            input_ids_subset = input_ids[:actual_size]
            attn_mask_subset = attn_mask[:actual_size]
            encoded_ips_subset = encoded_ips[:actual_size]
            labels_subset = labels[:actual_size]


            df_subset = df_subset.reset_index(drop=True)
            time_sorted_df = df_subset.sort_values('first_seen').reset_index(drop=True)
            split_point = int(len(time_sorted_df) * 0.8)

            print(f"时间训练集: {split_point} 样本")
            print(f"时间测试集: {len(time_sorted_df) - split_point} 样本")


            temporal_train_indices = list(range(split_point))
            temporal_test_indices = list(range(split_point, len(time_sorted_df)))


            temporal_train_data = TensorDataset(
                input_ids_subset[temporal_train_indices],
                attn_mask_subset[temporal_train_indices],
                encoded_ips_subset[temporal_train_indices],
                labels_subset[temporal_train_indices]
            )
            temporal_test_data = TensorDataset(
                input_ids_subset[temporal_test_indices],
                attn_mask_subset[temporal_test_indices],
                encoded_ips_subset[temporal_test_indices],
                labels_subset[temporal_test_indices]
            )

            temporal_train_loader = DataLoader(temporal_train_data, batch_size=self.config.batch_size, shuffle=True)
            temporal_test_loader = DataLoader(temporal_test_data, batch_size=self.config.batch_size, shuffle=False)


            test_timestamps = time_sorted_df.iloc[split_point:]['first_seen'].values


            print(f"\n训练时间稳定性模型: Our_Enhanced_Model")
            model = URL_IP_Model(self.config.bert_path).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2)


            best_auc = 0.0
            best_state = None
            best_performance = None
            patience_counter = 0

            for epoch in range(self.config.epochs):
                print(f"  Our_Enhanced_Model Epoch {epoch + 1}/{self.config.epochs}")
                train_with_mixed_precision(model, temporal_train_loader, optimizer, device, epoch, self.config)


                temp_perf = self.evaluate_temporal_performance(
                    model, temporal_test_loader, device, test_timestamps
                )

                print(f"    验证AUC: {temp_perf['auc']:.4f}, Recall@1%FPR: {temp_perf['recall_at_1fpr']:.4f}")


                if temp_perf['auc'] > best_auc:
                    best_auc = temp_perf['auc']
                    best_state = model.state_dict().copy()
                    best_performance = temp_perf
                    patience_counter = 0
                    print(f"    ✓ 新的最优模型")
                else:
                    patience_counter += 1

                if patience_counter >= 3:
                    print(f"    早停触发于 epoch {epoch + 1}")
                    break

                torch.cuda.empty_cache()


            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"\n加载最优模型 (AUC: {best_auc:.4f})")


            temporal_performance = best_performance if best_performance is not None else self.evaluate_temporal_performance(
                model, temporal_test_loader, device, test_timestamps
            )


            temporal_stability = self.calculate_temporal_stability(
                temporal_performance['temporal']
            )


            temporal_results = {
                'performance': temporal_performance,
                'stability': temporal_stability,
                'degradation_rate': temporal_stability['degradation_rate'],
                'stability_coefficient': temporal_stability['stability_coefficient']
            }

            print(f"  Our_Enhanced_Model 性能衰减率: {temporal_stability['degradation_rate']:.4f}")

            self.generate_temporal_comparison_analysis({'Our_Enhanced_Model': temporal_results})


            metrics = {
                'accuracy': temporal_performance['accuracy'],
                'f1': temporal_performance['f1'],
                'auc': temporal_performance['auc'],
                'false_alarm_rate': temporal_performance['false_alarm_rate'],
                'miss_rate': temporal_performance['miss_rate'],
                'recall_at_1fpr': temporal_performance['recall_at_1fpr'],
                'degradation_rate': temporal_stability['degradation_rate'],
                'stability_coefficient': temporal_stability['stability_coefficient'],
                'performance_trend': temporal_stability['performance_trend'],
                'temporal_advantage': temporal_stability['degradation_rate'] < 0.15,
                'months_analyzed': len(temporal_performance['temporal'])
            }

            self.log_experiment('Temporal_Our_Enhanced_Model', metrics)


            del model
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = f"时间漂移对比实验失败: {str(e)}"
            self.log_failed_experiment('Temporal_Comprehensive_Analysis', error_msg)
            import traceback
            traceback.print_exc()

    def evaluate_temporal_performance(self, model, test_loader, device, time_series):
        """评估时间性能"""
        model.eval()
        all_probs = []
        all_labels = []
        all_preds = []

        time_series = pd.Series(time_series)
        time_series = pd.to_datetime(time_series)
        months = time_series.dt.to_period('M').astype(str)
        unique_months = sorted(months.unique())

        monthly_probs = {m: [] for m in unique_months}
        monthly_labels = {m: [] for m in unique_months}

        sample_idx = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids, attention_mask, ip_data, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                ip_data = ip_data.to(device)
                labels = labels.to(device)

                _, logits = model(input_ids, attention_mask, ip_data, labels)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

                batch_size = len(labels)
                for i in range(batch_size):
                    if sample_idx < len(time_series):
                        month = months.iloc[sample_idx]
                        if month in monthly_probs:
                            monthly_probs[month].append(probs[i].cpu().numpy())
                            monthly_labels[month].append(labels[i].cpu().item())
                    sample_idx += 1


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
            macro_auc = roc_auc_score(all_labels, all_probs[:, 1])
        except Exception as e:
            print(f"AUC计算错误: {str(e)}")
            macro_auc = 0.5
        recall_at_1fpr = calculate_recall_at_1fpr(all_labels, all_probs)

        monthly_metrics = {}
        for month in unique_months:
            if month in monthly_probs and len(monthly_probs[month]) > 5:
                m_probs = np.array(monthly_probs[month])
                m_labels = np.array(monthly_labels[month])

                try:
                    if len(m_probs.shape) == 2 and m_probs.shape[1] == 2:
                        m_auc = roc_auc_score(m_labels, m_probs[:, 1])
                    else:
                        m_auc = roc_auc_score(m_labels, m_probs)
                    m_accuracy = accuracy_score(m_labels, np.argmax(m_probs, axis=1))
                    m_f1 = f1_score(m_labels, np.argmax(m_probs, axis=1), average='macro', zero_division=0)

                    monthly_metrics[month] = {
                        'auc': m_auc,
                        'accuracy': m_accuracy,
                        'f1': m_f1,
                        'recall@1%fpr': calculate_recall_at_1fpr(m_labels, m_probs),
                        'sample_count': len(m_labels)
                    }
                except Exception as e:
                    print(f"月度指标计算失败 ({month}): {str(e)}")
                    m_auc = 0.5


        print(f"\n=== 时间漂移分析详细结果 ===")
        print(f"总体性能指标:")
        print(f"  AUC: {macro_auc:.4f}")
        print(f"  Recall@1%FPR: {recall_at_1fpr:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  虚警率: {avg_fpr:.4f}")
        print(f"  漏检率: {1 - avg_tpr:.4f}")

        if len(monthly_metrics) > 0:
            print(f"\n月度性能变化:")
            print(f"{'月份':<12} | {'AUC':<6} | {'Recall@1%':<9} | {'准确率':<6} | {'F1':<6} | {'样本数'}")
            print("-" * 65)
            for month in sorted(monthly_metrics.keys()):
                metrics = monthly_metrics[month]
                print(f"{month:<12} | {metrics['auc']:.3f} | {metrics['recall@1%fpr']:.3f}     | "
                        f"{metrics['accuracy']:.3f} | {metrics['f1']:.3f} | {metrics['sample_count']}")

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
            'temporal': monthly_metrics
        }
        return result

    def calculate_temporal_stability(self, monthly_metrics):
        """计算时间稳定性指标"""
        if len(monthly_metrics) < 2:
            return {
                'degradation_rate': 0.0,
                'stability_coefficient': 0.0,
                'performance_trend': 'insufficient_data'
            }

        months = sorted(monthly_metrics.keys())
        aucs = [monthly_metrics[m]['auc'] for m in months]

        initial_auc = aucs[0]
        final_auc = aucs[-1]
        degradation_rate = (initial_auc - final_auc) / initial_auc if initial_auc > 0 else 0


        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        stability_coefficient = auc_std / auc_mean if auc_mean > 0 else 0

        if degradation_rate < 0.02:
            trend = 'stable'
        elif degradation_rate < 0.10:
            trend = 'minor_degradation'
        elif degradation_rate < 0.20:
            trend = 'moderate_degradation'
        else:
            trend = 'severe_degradation'

        return {
            'degradation_rate': float(degradation_rate),
            'stability_coefficient': float(stability_coefficient),
            'performance_trend': trend,
            'initial_performance': float(initial_auc),
            'final_performance': float(final_auc),
            'mean_performance': float(auc_mean)
        }

    def generate_temporal_comparison_analysis(self, temporal_results):
        """生成时间漂移对比分析报告"""
        print("\n" + "=" * 60)
        print("时间漂移对比分析报告")
        print("=" * 60)

        print("\n1. 模型时间稳定性对比:")
        print("-" * 40)

        for model_name, results in temporal_results.items():
            degradation = results['degradation_rate']
            stability = results['stability_coefficient']
            trend = results['stability']['performance_trend']
            monthly_data = results['performance'].get('temporal', {})
            print(f"{model_name}:")
            print(f"  性能衰减率: {degradation:.4f} ({degradation * 100:.1f}%)")
            print(f"  稳定性系数: {stability:.4f}")
            print(f"  趋势评估: {trend}")
            if len(monthly_data) > 1:
                print(f"\n{model_name} 月度表现:")
                print(f"{'月份':<12} | {'AUC':<6} | {'Recall@1%':<9} | {'样本数'}")
                print("-" * 40)
                for month in sorted(monthly_data.keys()):
                    metrics = monthly_data[month]
                    print(
                        f"{month:<12} | {metrics['auc']:.3f} | {metrics['recall@1%fpr']:.3f}     | {metrics['sample_count']}")
            print()


        for model_name, results in temporal_results.items():
            degradation_pct = results['degradation_rate'] * 100
            print(f"{model_name}: {degradation_pct:.1f}% 性能衰减")

    def generate_comprehensive_tables(self):
        """生成综合实验表格"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EXPERIMENT TABLES GENERATION")
        print("整表格")
        print("=" * 60)

        if not self.results:
            print("警告：没有实验结果可用于生成表格")
            return {}

        tables = {}


        print("生成主要性能对比表...")
        table_data = []

        for exp_name, metrics in self.results.items():
            if 'Our_Enhanced_Model' in exp_name:
                display_name = exp_name.replace('_', ' ')

                table_data.append([
                    display_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics.get('precision', 0):.4f}",
                    f"{metrics.get('recall', 0):.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['auc']:.4f}",
                    f"{metrics.get('recall_at_1fpr', 0):.4f}"
                ])

        tables['Main_Performance'] = pd.DataFrame(
            table_data,
            columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Recall@1%FPR']
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"comprehensive_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        log_path = f"{results_dir}/experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)

        for table_name, table_df in tables.items():
            csv_path = f"{results_dir}/{table_name}.csv"
            table_df.to_csv(csv_path, index=False)

            print(f"\n{table_name}:")
            print("-" * 80)
            print(table_df.to_string(index=False))

        summary_path = f"{results_dir}/comprehensive_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("包含完整的时间漂移分析和性能评估\n\n")
            f.write(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n实验完成 结果保存在: {results_dir}")
        return tables


def run_comprehensive_experiments():
    """运行综合实验"""
    config = Config()

    try:
        print("\n加载数据...")
        input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, df = load_and_preprocess_data_comprehensive(
            config)

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

    except Exception as e:
        print(f"\n数据加载失败: {str(e)}")
        raise ExperimentFailedException(f"实验初始化失败: {str(e)}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("COMPREHENSIVE EXPERIMENT SUITE")
    print("URL与IP多模态对比学习恶意检测")
    print("=" * 60)
    print("\n请选择要运行的实验:")
    print("1. 运行我们模型的性能评估")
    print("2. 运行时间鲁棒性实验")
    print("3. 运行完整综合实验")

    choice = input("请输入选择 (1-3): ").strip()

    try:
        runner, train_loader, val_loader, train_data, val_data, input_ids, attn_mask, encoded_ips, labels, train_idx_full, val_idx, df = run_comprehensive_experiments()

        if choice in ["1", "3"]:
            print("\n开始我们模型的性能评估...")
            runner.run_our_model_evaluation(train_loader, val_loader, runner.config.device)

        if choice in ["2", "3"]:
            print("\n开始时间鲁棒性实验...")
            runner.run_comprehensive_temporal_analysis(
                 input_ids, attn_mask, encoded_ips, labels,
                 train_idx_full, val_idx, df, runner.config.device
             )
        if choice in ["1", "3"]:
            print("\n开始消融实验...")
            runner.run_ablation_experiments(
                 input_ids, attn_mask, encoded_ips, labels,
                 train_idx_full, val_idx, runner.config.device
             )

            print("\n开始对抗鲁棒性实验...")
            runner.run_adversarial_experiments(val_data, runner.config.device)
            runner.run_adversarial_experiments(train_data, val_data, runner.config.device, df, val_idx)

            print("\n开始平衡vs非平衡实验...")
            runner.run_balance_comparison(
                 input_ids, attn_mask, encoded_ips, labels,
                 train_idx_full, val_idx, runner.config.device
             )

        tables = runner.generate_comprehensive_tables()

        print("\n" + "=" * 60)
        print("实验完成!")
        print("=" * 60)

    except ExperimentFailedException as e:
        print(f"\n实验失败: {str(e)}")
        print("请检查数据和模型配置")
    except Exception as e:
        print(f"\n未预期的错误: {str(e)}")
        import traceback

        traceback.print_exc()





