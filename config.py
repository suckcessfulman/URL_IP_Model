import os
import torch


class Config:
    """
    Central configuration for URL-IP malicious URL detection.

    raw_url_max_chars = 200: raw URL character-level preprocessing cap.
    max_url_tokens = 32: token-level sequence length after DistilBERT tokenization.
    """

    def __init__(self):
        self.batch_size = 64
        self.epochs = 12
        self.lr = 1e-4
        self.l2 = 1e-3
        self.num_workers = 0

        self.bert_path = "/root/autodl-tmp/url_detection/distilbert-base-uncased"

        # Two-stage URL length control.
        self.raw_url_max_chars = 200
        self.max_url_tokens = 32

        self.max_ttl = 86400
        self.max_ips = 3

        self.data_path = "/root"
        self.parquet_file = "urls_with_dns"
        self.data_file = os.path.join(self.data_path, self.parquet_file)

        # Use None for full data. Use 0.001 only for quick debugging.
        self.debug_sample_fraction = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = "./checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
