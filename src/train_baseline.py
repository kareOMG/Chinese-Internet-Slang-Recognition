import os
import joblib
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from src import config
from src import data_loader

def chinese_tokenizer(text):
    """
    使用 jieba 分词对中文句子进行分词，以空格连接
    """
    return list(jieba.cut(str(text)))

def train_baseline_model():
    print("\n" + "="*20 + " 1. 开始训练 TF-IDF + 逻辑回归 基线模型 " + "="*20)
    
    # 1. 下载和预处理数据
    df = data_loader.download_and_preprocess_data()
    
    # 2. 划分数据集
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    # 3. 特征工程：TF-IDF
    print("[特征工程] 正在对中文文本进行分词与 TF-IDF 向量化...")
    # 使用自定义分词器进行 TF-IDF 拟合
    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, max_features=10000, token_pattern=None)
    
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    # 4. 训练逻辑回归模型
    print("[模型训练] 正在训练 Logistic Regression 分类器...")
    model = LogisticRegression(random_state=config.SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 5. 评估模型
    print("[模型评估] 正在测试集上进行评估...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    
    print(f"\n[评估结果] TF-IDF + Logistic Regression 测试集指标:")
    print(f" - Accuracy:  {acc:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f" - F1-Score:  {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=["日常/标准 (Label 0)", "网络梗/抽象 (Label 1)"]))
    
    # 6. 保存模型与 Vectorizer
    model_data = {
        "vectorizer": vectorizer,
        "model": model
    }
    joblib.dump(model_data, config.BASELINE_WEIGHT_PATH)
    print(f"[模型保存] TF-IDF + LR 模型已保存至 {config.BASELINE_WEIGHT_PATH}")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    train_baseline_model()
