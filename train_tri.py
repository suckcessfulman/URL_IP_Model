import os
import time
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import DistilBertTokenizer

from config import Config
from data import load_data, encode_url, encode_ip_single, add_noise, check_url_token_consistency
from model_tri import URL_IP_Model

CLASS_NAMES = ["benign", "phishing", "malware"]


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().map({"benign": 0, "phishing": 1, "malware": 2, "malicious": 2, "mal": 2})
    df = df[df["label"].isin([0, 1, 2])]
    df["label"] = df["label"].astype(int)
    return df


def load_and_preprocess_data(config: Config):
    df = load_data(config.data_file)
    df = df.dropna(subset=["url", "ip_address"])
    assert "TTL" in df.columns, "Dataset is missing TTL column."
    assert "label" in df.columns, "Dataset is missing label column."
    df = map_labels(df)

    if config.debug_sample_fraction is not None:
        _, df = train_test_split(df, test_size=config.debug_sample_fraction, stratify=df["label"], random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained(config.bert_path)
    urls = df["url"].tolist()
    urls = [add_noise(url) if random.random() < 0.5 else url for url in urls]
    encoded_urls = encode_url(urls, tokenizer, max_length=config.max_url_tokens, raw_max_chars=config.raw_url_max_chars)

    try:
        check_result = check_url_token_consistency(df["url"].head(5000).tolist(), tokenizer, max_length=config.max_url_tokens, raw_max_chars=config.raw_url_max_chars)
        print(f"URL 200-char cap vs 32-token consistency on sample: {check_result}")
    except Exception as exc:
        print(f"Skipped URL consistency check due to: {exc}")

    encoded_ips = []
    for ips, ttl in tqdm(zip(df["ip_address"], df["TTL"]), total=len(df), desc="Encoding IPs"):
        encoded_ips.append(encode_ip_single(ips, ttl, max_ips=config.max_ips, max_ttl=config.max_ttl))
    encoded_ips = torch.stack(encoded_ips)

    input_ids = encoded_urls["input_ids"]
    attention_mask = encoded_urls["attention_mask"]
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    assert not torch.isnan(encoded_ips).any(), "Encoded IPs contain NaN values."

    print("Label distribution:")
    print(df["label"].value_counts().sort_index())

    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels.numpy(), random_state=42)
    timestamps = df["first_seen"].values if "first_seen" in df.columns else None
    return input_ids, attention_mask, encoded_ips, labels, train_idx, val_idx, timestamps


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, total_epochs, scheduler=None):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", ncols=100, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        input_ids, attention_mask, ip_data, labels = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, ip_data=ip_data, labels=labels, current_epoch=epoch, total_epochs=total_epochs)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        pbar.set_postfix(loss=total_loss / (batch_idx + 1), acc=total_correct / total_count)
    return total_loss / len(train_loader), total_correct / total_count


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for batch in tqdm(data_loader, desc="Evaluating", ncols=100, dynamic_ncols=True):
        input_ids, attention_mask, ip_data, labels = [x.to(device) for x in batch]
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask, ip_data=ip_data, labels=labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.asarray(all_probs)
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    return {
        "auc_macro_ovo": roc_auc_score(all_labels, all_probs, multi_class="ovo", average="macro") if len(np.unique(all_labels)) == 3 else float("nan"),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "report": classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0),
    }


def main():
    config = Config()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    input_ids, attn_mask, encoded_ips, labels, train_idx, val_idx, _ = load_and_preprocess_data(config)

    tensors = [input_ids, attn_mask, encoded_ips, labels]
    train_data = TensorDataset(*[t[train_idx] for t in tensors])
    val_data = TensorDataset(*[t[val_idx] for t in tensors])
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=config.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=config.num_workers)

    model = URL_IP_Model(config.bert_path).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    best_score = -1.0
    for epoch in range(config.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, config.device, epoch, config.epochs, scheduler)
        metrics = evaluate(model, val_loader, config.device)
        score = metrics["macro_f1"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_tri_model.pt"))
        print(f"Epoch {epoch + 1:02d}/{config.epochs} | Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Macro-F1 {metrics["macro_f1"]:.4f} | Acc {metrics["accuracy"]:.4f} | Macro-AUC(OVO) {metrics["auc_macro_ovo"]:.4f} | Time {time.time() - start:.1f}s")
        if "report" in metrics:
            print(metrics["report"])
    print(f"Best validation macro_f1: {best_score:.4f}")


if __name__ == "__main__":
    main()
