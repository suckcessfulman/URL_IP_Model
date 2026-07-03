import os
import glob
import re
import random
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import torch


def load_data(file_path: str) -> pd.DataFrame:
    """Load parquet data from a single parquet file or a directory of parquet files."""
    print(f"Trying to load data from: {file_path}")

    if os.path.isfile(file_path):
        print(f"Detected single file: {file_path}")
        df = pd.read_parquet(file_path)
    elif os.path.isdir(file_path):
        print(f"Detected directory: {file_path}")
        parquet_files = sorted(glob.glob(os.path.join(file_path, "*.parquet")))
        if not parquet_files:
            print(f"Directory contents: {os.listdir(file_path)}")
            raise FileNotFoundError(f"No parquet files found in {file_path}")
        dfs = []
        for file in parquet_files:
            print(f"Loading: {file}")
            dfs.append(pd.read_parquet(file))
        df = pd.concat(dfs, ignore_index=True)
    else:
        parent_dir = os.path.dirname(file_path)
        if parent_dir and os.path.exists(parent_dir):
            print(f"Parent directory {parent_dir} exists. Contents: {os.listdir(parent_dir)}")
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    if "first_seen" in df.columns:
        df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")

    print(f"Loaded {len(df)} records.")
    return df


def normalize_url(url: Any, max_chars: int = 200) -> str:
    """
    Raw URL preprocessing.

    This implements the manuscript statement: "caps URL length at 200 characters".
    DistilBERT tokenization later uses max_url_tokens=32.
    """
    if pd.isna(url):
        return ""
    url = str(url)
    url = url.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    url = url.strip()
    return url[:max_chars]


def encode_url(urls: Iterable[Any], tokenizer, max_length: int = 32, raw_max_chars: int = 200):
    """
    Two-stage length control:
    1) raw URL cap: 200 characters;
    2) DistilBERT token sequence cap: 32 tokens.
    """
    urls = [normalize_url(url, max_chars=raw_max_chars) for url in urls]
    return tokenizer.batch_encode_plus(
        urls,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )


def _to_list_maybe(value: Any) -> List[Any]:
    """Convert scalar/list/ndarray/string-stored list to a Python list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return list(parsed)
            except Exception:
                pass
        if "," in s:
            return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]
        if ";" in s:
            return [x.strip().strip("'\"") for x in s.split(";") if x.strip()]
        return [s]
    return [value]


def encode_ip_single(ip_entries: Any, ttl_entry: Any, max_ips: int = 3, default: float = 0.0, max_ttl: int = 86400) -> torch.Tensor:
    """
    Encode up to 3 IPv4 addresses and TTLs into 21 dimensions:
    3 * (4 octets + 1 TTL) + valid_count_ratio + avg_4_octets + avg_TTL = 21.
    """
    pattern = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
    ip_list = _to_list_maybe(ip_entries)
    ttl_list = _to_list_maybe(ttl_entry)

    if not ttl_list:
        ttl_list = [default] * len(ip_list)
    elif len(ttl_list) == 1 and len(ip_list) > 1:
        ttl_list = ttl_list * len(ip_list)
    elif len(ttl_list) < len(ip_list):
        ttl_list = ttl_list + [default] * (len(ip_list) - len(ttl_list))

    features = []
    valid_count = 0
    valid_ips = []
    valid_ttls = []

    for ip, ttl in zip(ip_list[:max_ips], ttl_list[:max_ips]):
        ip = str(ip).strip().strip("'\"")
        try:
            ttl_val = float(ttl)
            match = pattern.match(ip)
            if not match:
                raise ValueError("Invalid IPv4 pattern")
            octet_ints = [int(g) for g in match.groups()]
            if not all(0 <= x <= 255 for x in octet_ints):
                raise ValueError("IPv4 octet out of range")
            if not (0 <= ttl_val <= max_ttl):
                raise ValueError("TTL out of range")
            octets = [x / 255.0 for x in octet_ints]
            norm_ttl = ttl_val / max_ttl
            features.extend(octets + [norm_ttl])
            valid_ips.append(octets)
            valid_ttls.append(norm_ttl)
            valid_count += 1
        except Exception:
            features.extend([default] * 5)

    while len(features) < max_ips * 5:
        features.extend([default] * 5)

    count_ratio = valid_count / max_ips
    avg_octets = torch.mean(torch.tensor(valid_ips, dtype=torch.float32), dim=0) if valid_ips else torch.zeros(4, dtype=torch.float32)
    avg_ttl = torch.mean(torch.tensor(valid_ttls, dtype=torch.float32)) if valid_ttls else torch.tensor(0.0, dtype=torch.float32)

    return torch.cat([
        torch.tensor(features, dtype=torch.float32),
        torch.tensor([count_ratio], dtype=torch.float32),
        avg_octets,
        avg_ttl.unsqueeze(0),
    ])


def add_noise(url: Any, noise_prob: float = 0.1) -> str:
    url = "" if pd.isna(url) else str(url)
    chars = list(url)
    i = 0
    while i < len(chars):
        if random.random() < noise_prob:
            op = random.choice(["replace", "delete", "insert"])
            if op == "replace":
                chars[i] = random.choice(["x", "0", "_"])
            elif op == "delete":
                chars[i] = ""
            elif op == "insert":
                chars.insert(i, random.choice(["-", "."]))
                i += 1
        i += 1
    return "".join(chars)


def check_url_token_consistency(urls: Iterable[Any], tokenizer, max_length: int = 32, raw_max_chars: int = 200):
    """Optional sanity check: compare tokenized inputs with/without the 200-character cap."""
    same, total = 0, 0
    diff_examples = []
    for url in urls:
        a = tokenizer("" if pd.isna(url) else str(url), max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"][0]
        b = tokenizer(normalize_url(url, max_chars=raw_max_chars), max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"][0]
        total += 1
        if torch.equal(a, b):
            same += 1
        elif len(diff_examples) < 5:
            diff_examples.append(str(url))
    return {"same": same, "total": total, "same_ratio": same / total if total else 0.0, "diff_examples": diff_examples}
