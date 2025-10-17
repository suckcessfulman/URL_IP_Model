import torch
import os
# 保持ori_config.py的简洁结构，只修改必要的路径和参数
class Config:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 12
        self.bert_path = "/root/autodl-tmp/url_detection/distilbert-base-uncased"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lr = 1e-4
        self.l2 = 1e-3
        self.data_path = "/root"
        self.parquet_file = "urls_with_dns"
        self.max_ttl = 86400
        self.oversample = True  # 只保留这一个有用的参数