import os
# 设置 Hugging Face 镜像源以确保国内网络下载成功
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from src import config

def download_and_preprocess_data():
    """
    下载数据集并进行数据清洗和二分类样本构建。
    """
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"[数据加载] 发现已处理好的本地数据集: {config.RAW_DATA_PATH}，直接加载。")
        return pd.read_csv(config.RAW_DATA_PATH)

    print("[数据下载] 正在从 Hugging Face 下载数据集 'GaryYang123/zh-meme-sft-8k'...")
    try:
        # 加载数据集
        dataset = load_dataset("GaryYang123/zh-meme-sft-8k")
        # 默认只有 train 分区
        data_list = dataset['train']
    except Exception as e:
        print(f"[错误] 数据集下载失败: {e}")
        print("[回退提示] 将自动生成模拟数据集以确保项目能够正常运行。")
        return generate_mock_data()

    texts = []
    labels = []

    print("[数据清洗] 正在解析 ChatML 格式数据并构建二分类数据集...")
    for item in tqdm(data_list):
        # 兼容不同的数据结构，有些数据集是 'messages'，有些是直接的键值
        messages = item.get("messages", [])
        if not messages:
            continue
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if not content:
                continue
            
            # Label 0: 标准/常规提问 (来自 user)
            # Label 1: 幽默/网络梗/抽象回复 (来自 assistant)
            if role == "user":
                texts.append(content)
                labels.append(0)
            elif role == "assistant":
                texts.append(content)
                labels.append(1)

    df = pd.DataFrame({"text": texts, "label": labels})
    
    # 简单清洗
    df = df.dropna().drop_duplicates(subset=["text"])
    # 过滤掉过短的文本
    df = df[df["text"].str.len() >= 2]
    
    # 数据集类别均衡检查与截断
    label_counts = df["label"].value_counts()
    print(f"[数据分析] 原始类别分布:\n{label_counts}")
    
    min_count = label_counts.min()
    df_class_0 = df[df["label"] == 0].sample(n=min_count, random_state=config.SEED)
    df_class_1 = df[df["label"] == 1].sample(n=min_count, random_state=config.SEED)
    df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    
    print(f"[数据分析] 均衡后类别分布:\n{df_balanced['label'].value_counts()}")
    
    # 保存至本地 CSV
    df_balanced.to_csv(config.RAW_DATA_PATH, index=False, encoding="utf-8")
    print(f"[数据清洗] 处理完成，共 {len(df_balanced)} 条样本。数据已保存至 {config.RAW_DATA_PATH}")
    return df_balanced

def generate_mock_data():
    """
    当网络连接完全受限时使用的替代模拟数据集，防止流程中断。
    """
    print("[数据加载] 正在生成中文互联网梗与标准文本的模拟数据集...")
    mock_user = [
        "今天天气真好，我们去公园散步吧。",
        "请问这个数学公式应该怎么推导？",
        "下周我要交三份期末报告，时间有点紧。",
        "人工智能未来的发展趋势是什么？",
        "如何做出一碗美味的红烧肉？",
        "请推荐几本适合夏天阅读的小说。",
        "学习Python编程需要哪些基础知识？",
        "如何提高日常工作的效率？",
        "中国的高铁系统在世界上处于什么水平？",
        "地球和太阳的距离是多少公里？"
    ] * 500  # 复制多次以获得足够数据
    
    mock_assistant = [
        "我去不早说，这谁顶得住啊！",
        "老铁，你这算盘珠子都崩我脸上了。",
        "神仙打架，我直接一个好家伙，奥利给！",
        "尊嘟假嘟？我觉得你是在无中生有暗度陈仓。",
        "当代大学生现状：恋爱可以不谈，但期末考试不能挂。",
        "小丑竟是我自己，我直接原地退网。",
        "这波操作直接在大气层，我给跪了。",
        "兄弟，建议直接买张站票连夜逃跑。",
        "针不戳，这小日子过得针不戳！",
        "伤害性不大，侮辱性极强，我直接一整个大无语。"
    ] * 500
    
    texts = mock_user + mock_assistant
    labels = [0] * len(mock_user) + [1] * len(mock_assistant)
    
    df = pd.DataFrame({"text": texts, "label": labels}).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df.to_csv(config.RAW_DATA_PATH, index=False, encoding="utf-8")
    return df

# ==================== BiLSTM 专用词表与 Dataset ====================

class Vocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def build_vocab(self, texts, vocab_size):
        word_counts = {}
        for text in texts:
            words = list(jieba.cut(text))
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 按照词频排序
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        # 限制词表大小
        for word, _ in sorted_words[:vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def text_to_indices(self, text, max_len):
        words = list(jieba.cut(text))
        indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(indices) < max_len:
            indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices
    
    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": {str(k): v for k, v in self.idx2word.items()}}, f, ensure_ascii=False)
            
    @classmethod
    def load(cls, filepath):
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab.word2idx = data["word2idx"]
            vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return vocab


class BiLSTMDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.labels = df["label"].values
        self.features = []
        for text in df["text"].values:
            self.features.append(vocab.text_to_indices(text, max_len))
        self.features = np.array(self.features)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# ==================== BERT 专用 Dataset ====================

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ==================== 数据划分函数 ====================

def get_train_val_test_splits(df):
    """
    按 8:1:1 比例划分训练集、验证集和测试集。
    """
    # 第一次划分：划分出训练集占 80%，临时集占 20%
    train_df, temp_df = train_test_split(
        df, test_size=(config.VAL_RATIO + config.TEST_RATIO), random_state=config.SEED, stratify=df["label"]
    )
    # 第二次划分：将临时集平分，即各占总体的 10%
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=config.SEED, stratify=temp_df["label"]
    )
    
    print(f"[数据划分] 划分完成: 训练集={len(train_df)}，验证集={len(val_df)}，测试集={len(test_df)}")
    return train_df, val_df, test_df
