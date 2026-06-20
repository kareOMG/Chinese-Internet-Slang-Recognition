import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

from src import config
from src import data_loader
from src import models

def train_bilstm_model():
    print("\n" + "="*20 + " 2. 开始训练 BiLSTM + Attention 深度学习模型 " + "="*20)
    print(f"[设备检测] 当前训练设备: {config.DEVICE}")
    
    # 1. 准备数据
    df = data_loader.download_and_preprocess_data()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    # 2. 构建词表并保存
    vocab_path = os.path.join(config.OUTPUT_DIR, "bilstm_vocab.json")
    print("[词表构建] 正在基于训练集语料构建分词词表...")
    vocab = data_loader.Vocab()
    vocab.build_vocab(train_df["text"].values, config.VOCAB_SIZE)
    vocab.save(vocab_path)
    print(f"[词表构建] 词表大小: {len(vocab.word2idx)}，已保存至 {vocab_path}")
    
    # 3. 创建 PyTorch Dataset 与 DataLoader
    print("[数据准备] 正在将文本转换为 Token 序列并构建 DataLoader...")
    train_dataset = data_loader.BiLSTMDataset(train_df, vocab, config.LSTM_MAX_LEN)
    val_dataset = data_loader.BiLSTMDataset(val_df, vocab, config.LSTM_MAX_LEN)
    test_dataset = data_loader.BiLSTMDataset(test_df, vocab, config.LSTM_MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型、损失函数与优化器
    model = models.BiLSTMAttention(
        vocab_size=len(vocab.word2idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LSTM_LR)
    
    # 5. 训练循环
    best_val_acc = 0.0
    
    # 保存历史指标以便于可视化
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print(f"[训练启动] 计划训练 {config.LSTM_EPOCHS} 个 Epoch...")
    for epoch in range(config.LSTM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 训练批次循环
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.LSTM_EPOCHS} [Train]"):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE, dtype=torch.long)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.LSTM_EPOCHS} [Val]"):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE, dtype=torch.long)
                
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # 记录历史
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
        
        # 如果模型在验证集上表现更好，保存模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), config.LSTM_WEIGHT_PATH)
            print(f" -> 验证集表现提升，模型已保存至 {config.LSTM_WEIGHT_PATH}")
            
    # 6. 在测试集上做最终评估
    print("\n[加载最优模型] 正在加载保存的最佳模型权重并在测试集上进行最终评估...")
    best_model = models.BiLSTMAttention(
        vocab_size=len(vocab.word2idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    best_model.load_state_dict(torch.load(config.LSTM_WEIGHT_PATH, map_location=config.DEVICE))
    best_model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            logits, _ = best_model(inputs)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
    
    print(f"\n[评估结果] BiLSTM + Attention 测试集指标:")
    print(f" - Accuracy:  {test_acc:.4f}")
    print(f" - Precision: {test_precision:.4f}")
    print(f" - Recall:    {test_recall:.4f}")
    print(f" - F1-Score:  {test_f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_preds, target_names=["日常/标准 (Label 0)", "网络梗/抽象 (Label 1)"]))
    
    # 保存训练历史用于绘图
    history_path = os.path.join(config.OUTPUT_DIR, "bilstm_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
        
    return {
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1
    }

if __name__ == "__main__":
    train_bilstm_model()
