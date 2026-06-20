import os
import torch

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 创建必要目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 缓存与数据路径
RAW_DATA_PATH = os.path.join(DATA_DIR, "zh_meme_sft_8k_processed.csv")

# 随机种子，确保实验可重复
SEED = 42

# 训练设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ==================== 模型超参数 ====================

# 1. BiLSTM + Attention
VOCAB_SIZE = 15000       # 词表最大容量
EMBEDDING_DIM = 128     # 词向量维度
HIDDEN_DIM = 128        # LSTM隐藏层维度
NUM_LAYERS = 2          # LSTM层数
DROPOUT = 0.3           # Dropout比例
LSTM_LR = 1e-3          # BiLSTM学习率
LSTM_BATCH_SIZE = 64    # BiLSTM批次大小
LSTM_EPOCHS = 10        # BiLSTM训练轮数
LSTM_MAX_LEN = 64       # BiLSTM最大句子长度
LSTM_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "bilstm_attention.pth")

# 2. BERT Text Classification
BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 性能优秀的中文RoBERTa模型，备用: bert-base-chinese
BERT_LR = 2e-5                                  # BERT微调学习率
BERT_BATCH_SIZE = 32                            # BERT批次大小
BERT_EPOCHS = 3                                 # BERT训练轮数
BERT_MAX_LEN = 64                               # BERT最大句子长度
BERT_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "bert_classifier.pth")
BERT_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")

# 3. TF-IDF + Logistic Regression
BASELINE_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "tfidf_lr.pkl")
