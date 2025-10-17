import pandas as pd
import torch
from transformers import DistilBertTokenizer
import random
import torch.nn as nn
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import os
import glob

# 增强版加载数据函数，跨平台兼容
def load_data(file_path):
    """
    加载parquet数据，支持单文件和文件夹两种形式
    兼容Windows和Linux路径
    """
    print(f"正在尝试加载数据: {file_path}")
    
    if os.path.isfile(file_path):
        # 如果是单个文件
        print(f"检测到单个文件: {file_path}")
        df = pd.read_parquet(file_path)
    elif os.path.isdir(file_path):
        # 如果是文件夹，读取所有parquet文件
        print(f"检测到文件夹: {file_path}")
        parquet_files = glob.glob(os.path.join(file_path, "*.parquet"))
        if not parquet_files:
            # 尝试查找其他可能的文件格式
            print(f"未找到.parquet文件，正在查找文件夹内容...")
            all_files = os.listdir(file_path)
            print(f"文件夹内容: {all_files}")
            raise FileNotFoundError(f"在 {file_path} 中未找到parquet文件")
        
        print(f"找到 {len(parquet_files)} 个parquet文件: {parquet_files}")
        dfs = []
        for file in parquet_files:
            print(f"正在加载: {file}")
            dfs.append(pd.read_parquet(file))
        df = pd.concat(dfs, ignore_index=True)
    else:
        # 路径不存在，显示更多调试信息
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir):
            print(f"父目录 {parent_dir} 存在，内容: {os.listdir(parent_dir)}")
        raise FileNotFoundError(f"路径不存在: {file_path}")

    # 假设原始数据包含first_seen字段
    if 'first_seen' in df.columns:
        df['first_seen'] = pd.to_datetime(df['first_seen'])
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    return df

# 文本和 IP 的编码
def encode_url(urls, tokenizer, max_length=64):
    return tokenizer.batch_encode_plus(
        urls,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        return_token_type_ids=False  # 不需要 token_type_ids
    )


def ip_to_tensor(ip):
    try:
        # 去除可能的空格
        ip = ip.strip()

        # 如果有不合法字符，直接跳过
        if not ip or ip.count('.') != 3:
            raise ValueError(f"Invalid IP address format: {ip}")

        # 将 IP 地址按 '.' 分割
        parts = ip.split('.')

        # 确保 IP 地址格式正确，应该有四个部分
        if len(parts) != 4:
            raise ValueError(f"Invalid IP address format: {ip}")

        # 将每个部分转换为整数并归一化到 0-1 的范围
        parts = [int(part) / 255.0 for part in parts]

        # 将归一化的 IP 地址转为张量
        return torch.tensor(parts, dtype=torch.float32)
    except Exception as e:
        print(f"IP 地址转换错误: {ip}，错误信息: {str(e)}")
        return torch.zeros(4)  # 使用默认值填充（4维的零张量）

import re

# data.py修改部分
def encode_ip_single(ip_entries, ttl_entry, max_ips=3, default=0.0, max_ttl=86400):
    pattern = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
    features = []
    valid_count = 0
    valid_ips = []
    valid_ttls = []

    # 统一处理TTL输入格式（标量转列表）
    if not isinstance(ttl_entry, (list, np.ndarray)):
        ttl_entries = [ttl_entry] * len(ip_entries)  # 标量扩展为与IP等长列表
    else:
        ttl_entries = ttl_entry

    # 确保IP和TTL列表长度一致
    ttl_entries = list(ttl_entries) + [default] * (len(ip_entries) - len(ttl_entries))

    # 同步遍历处理逻辑
    for ip, ttl in zip(ip_entries[:max_ips], ttl_entries[:max_ips]):
        ip = str(ip).strip()
        try:
            ttl_val = float(ttl)  # 强制类型转换
            match = pattern.match(ip)
            if match and 0 <= ttl_val <= max_ttl:
                octets = [int(g) / 255.0 for g in match.groups()]
                norm_ttl = ttl_val / max_ttl
                features.extend(octets + [norm_ttl])
                valid_ips.append(octets)
                valid_ttls.append(norm_ttl)
                valid_count += 1
            else:
                features.extend([default] * 5)
        except (ValueError, AttributeError, TypeError):
            features.extend([default] * 5)

    # 填充剩余位置（保持特征维度稳定）
    features.extend([default] * 5 * (max_ips - valid_count))

    # 统计特征计算优化
    stats = [valid_count / max_ips]
    avg_octets = torch.mean(torch.tensor(valid_ips, dtype=torch.float32), dim=0) if valid_ips else torch.zeros(4)
    avg_ttl = torch.mean(torch.tensor(valid_ttls)) if valid_ttls else torch.tensor(0.0)

    return torch.cat([
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(stats, dtype=torch.float32),
        avg_octets,
        avg_ttl.unsqueeze(0)
    ])

def add_noise(url, noise_prob=0.1):
    chars = list(url)
    for i in range(len(chars)):
        if random.random() < noise_prob:
            # 随机选择扰动方式：替换/删除/插入
            op = random.choice(['replace','delete','insert'])
            if op == 'replace':
                chars[i] = random.choice(['x','0','_'])  # 用易混淆字符替换
            elif op == 'delete':
                chars[i] = ''
            elif op == 'insert':
                chars.insert(i, random.choice(['-','.']))
    return ''.join(chars)


def oversample_data(input_ids, attn_mask, encoded_ips, labels, indices):
    """
    二分类版本的过采样函数
    只处理两个类别: 0 (benign) 和 1 (malicious)
    """
    # 首先获取指定索引的数据
    X_input_ids = input_ids[indices]
    X_attn_mask = attn_mask[indices]
    X_encoded_ips = encoded_ips[indices]
    y = labels[indices]

    # 如果是PyTorch张量，先转换为numpy数组
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = y

    # 找出样本最多的类别数量（二分类：类别0和1）
    max_samples = max([np.sum(y_np == i) for i in range(2)])

    # 存储过采样后的数据
    oversampled_input_ids = []
    oversampled_attn_mask = []
    oversampled_encoded_ips = []
    oversampled_y = []

    # 对每个类别进行过采样（二分类：0和1）
    for i in range(2):
        # 使用numpy操作找出索引
        idx_np = np.where(y_np == i)[0]
        if len(idx_np) > 0:  # 确保该类别有样本
            # 计算需要重复的次数
            repeat_times = max_samples // len(idx_np)
            remainder = max_samples % len(idx_np)

            # 重复整数倍样本
            for _ in range(repeat_times):
                oversampled_input_ids.extend([X_input_ids[j] for j in idx_np])
                oversampled_attn_mask.extend([X_attn_mask[j] for j in idx_np])
                oversampled_encoded_ips.extend([X_encoded_ips[j] for j in idx_np])
                oversampled_y.extend([y[j] for j in idx_np])

            # 添加余数部分样本
            if remainder > 0:
                extra_idx_np = np.random.choice(idx_np, size=remainder, replace=False)
                oversampled_input_ids.extend([X_input_ids[j] for j in extra_idx_np])
                oversampled_attn_mask.extend([X_attn_mask[j] for j in extra_idx_np])
                oversampled_encoded_ips.extend([X_encoded_ips[j] for j in extra_idx_np])
                oversampled_y.extend([y[j] for j in extra_idx_np])

    # 将列表转换为适当的数据类型
    if torch.is_tensor(input_ids):
        return (torch.stack(oversampled_input_ids),
                torch.stack(oversampled_attn_mask),
                torch.stack(oversampled_encoded_ips),
                torch.tensor(oversampled_y))
    else:
        # 如果是numpy数组，使用np.array转换
        return (np.array(oversampled_input_ids),
                np.array(oversampled_attn_mask),
                np.array(oversampled_encoded_ips),
                np.array(oversampled_y))