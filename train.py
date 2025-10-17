import pandas as pd
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import URL_IP_Model
from data import load_data, encode_url, encode_ip_single, add_noise
from config import Config
import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset
import random
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

scaler = GradScaler()

from tqdm import tqdm  # Ensure tqdm is imported


def load_and_preprocess_data(config):
    # Load data
    df = load_data(config.data_path + "\\" + config.parquet_file)
    df = df.dropna(subset=['url', 'ip_address'])
    assert 'TTL' in df.columns, "数据集缺少TTL字段"

    # Preprocess labels
    df['label'] = df['label'].str.lower().map({
        'benign': 0,
        'malware': 1,
        'phishing': 1
    })
    df = df[df['label'].isin([0, 1])]
    df['label'] = df['label'].astype(int)

    # Train-test split
    from sklearn.model_selection import train_test_split
    _, df = train_test_split(df, test_size=0.001, stratify=df['label'], random_state=42)

    # Preprocess IP data with tqdm for progress
    encoded_ips = []
    for ips, ttl in tqdm(zip(df['ip_address'], df['TTL']), total=len(df), desc="Encoding IPs"):
        encoded = encode_ip_single(ips, ttl)
        encoded_ips.append(encoded)
    encoded_ips = torch.stack(encoded_ips)

    # Process URLs with noise
    urls = df['url'].tolist()
    urls = [add_noise(url) if random.random() < 0.5 else url for url in urls]

    tokenizer = DistilBertTokenizer.from_pretrained(config.bert_path)
    encoded_urls = encode_url(urls, tokenizer, max_length=32)

    input_ids = encoded_urls['input_ids']
    attention_mask = encoded_urls['attention_mask']

    labels = torch.tensor(df['label'].values, dtype=torch.long)

    assert not torch.isnan(encoded_ips).any(), "Encoded IPs contain NaN values!"
    print("Label distribution:", df['label'].value_counts())

    # Train-validation split
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=42
    )

    return (input_ids, attention_mask, encoded_ips, labels, train_idx, val_idx)

# Training loop with tqdm progress bar
def train_with_mixed_precision(model, train_loader, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0
    corrects = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", ncols=100, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
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
                labels=labels
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


def evaluate_with_metrics(model, test_loader, device, time_series=None, verbose=False):
    model.eval()
    all_probs = []
    all_labels = []
    time_metrics = {}

    # 时间处理逻辑
    if time_series is not None:
        time_series = pd.Series(time_series).copy()
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        months = time_series.dt.to_period('M').astype(str)
        unique_months = sorted(months.unique())
        monthly_probs = {m: [] for m in unique_months}
        monthly_labels = {m: [] for m in unique_months}
    else:
        unique_months = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, ip_data, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ip_data = ip_data.to(device)
            labels = labels.to(device)

            _, logits = model(input_ids, attention_mask, ip_data, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if time_series is not None:
                batch_start = batch_idx * test_loader.batch_size
                batch_end = (batch_idx + 1) * test_loader.batch_size
                batch_months = months.iloc[batch_start:batch_end]

                for i, m in enumerate(batch_months):
                    monthly_probs[m].append(probs[i].item())
                    monthly_labels[m].append(labels[i].item())

    # 全局指标
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    idx = np.where(fpr <= 0.01)[0][-1] if len(np.where(fpr <= 0.01)[0]) > 0 else 0
    recall_at_1fpr = tpr[idx]

    # 时间分段指标（仅当verbose=True时详细输出）
    if time_series is not None:
        for month in unique_months:
            m_probs = monthly_probs[month]
            m_labels = monthly_labels[month]
            if len(set(m_labels)) < 2:
                continue

            m_auc = roc_auc_score(m_labels, m_probs)
            m_fpr, m_tpr, _ = roc_curve(m_labels, m_probs)
            m_idx = np.where(m_fpr <= 0.01)[0][-1] if len(np.where(m_fpr <= 0.01)[0]) > 0 else 0
            m_recall = m_tpr[m_idx]

            time_metrics[month] = {
                'auc': m_auc,
                'recall@1%fpr': m_recall,
                'sample_count': len(m_labels)
            }

        # 新增：汇总统计（当verbose=True时显示）
        if verbose:
            print("\n=== Temporal Performance Summary ===")
            print(f"{'Period':<10} | {'AUC':<6} | {'Recall@1%':<9} | Samples")
            print("-" * 45)
            for month, metrics in sorted(time_metrics.items()):
                print(
                    f"{month:<10} | {metrics['auc']:.3f} | {metrics['recall@1%fpr']:.3f}     | {metrics['sample_count']}")

    return {
        'auc': auc,
        'recall@1%fpr': recall_at_1fpr,
        'temporal': time_metrics
    }


if __name__ == "__main__":
    config = Config()
    tokenizer = DistilBertTokenizer.from_pretrained(config.bert_path)

    # 加载数据
    input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx = load_and_preprocess_data(config)
    df = pd.read_parquet(config.data_path + "\\" + config.parquet_file)
    train_timestamps = df.iloc[train_idx]['first_seen'].values
    val_timestamps = df.iloc[val_idx]['first_seen'].values

    # 创建数据集和数据加载器
    all_tensors = [input_ids, attn_mask, encoded_ips, labels]
    train_data = TensorDataset(*[t[train_idx] for t in all_tensors])
    val_data = TensorDataset(*[t[val_idx] for t in all_tensors])
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # 初始化模型、优化器和学习率调度器
    model = URL_IP_Model(config.bert_path).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # 训练循环
    best_auc = 0.0
    temporal_history = []  # 存储时间维度指标

    for epoch in range(config.epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_acc = train_with_mixed_precision(
            model, train_loader, optimizer, config.device, epoch, scheduler
        )

        # 验证阶段
        val_metrics = evaluate_with_metrics(
            model, val_loader, config.device, val_timestamps, verbose=False
        )
        temporal_history.append(val_metrics['temporal'])  # 存储时间维度数据

        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pt")
            best_auc = val_metrics['auc']

        # 输出当前epoch结果
        print(f"Epoch {epoch + 1:2d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | Recall@1%: {val_metrics['recall@1%fpr']:.4f} | "
              f"Time: {time.time() - start_time:.1f}s")

    # 全部训练完成后显示时间趋势分析
    print("\n=== Final Temporal Analysis ===")
    final_temporal = temporal_history[-1]  # 取最后一个epoch的时间指标
    print(f"{'Period':<10} | {'AUC':<6} | {'Recall@1%':<9} | Samples")
    print("-" * 45)
    for month in sorted(final_temporal.keys()):
        metrics = final_temporal[month]
        print(f"{month:<10} | {metrics['auc']:.3f} | {metrics['recall@1%fpr']:.3f}     | {metrics['sample_count']}")