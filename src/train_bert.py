import os
# 设置 Hugging Face 镜像源以确保国内网络下载模型成功
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src import config
from src import data_loader

def train_bert_model():
    print("\n" + "="*20 + " 3. 开始微调 BERT (RoBERTa) 预训练模型 " + "="*20)
    print(f"[设备检测] 当前训练设备: {config.DEVICE}")
    
    # 1. 准备数据
    df = data_loader.download_and_preprocess_data()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    # 2. 初始化 Tokenizer
    print(f"[模型加载] 正在加载 AutoTokenizer: {config.BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BERT_MODEL_NAME, 
        cache_dir=config.BERT_CACHE_DIR
    )
    
    # 3. 创建 PyTorch Dataset 与 DataLoader
    print("[数据准备] 正在对中文文本进行 Token 编码并构建 BERT Dataset...")
    train_dataset = data_loader.BERTDataset(train_df, tokenizer, config.BERT_MAX_LEN)
    val_dataset = data_loader.BERTDataset(val_df, tokenizer, config.BERT_MAX_LEN)
    test_dataset = data_loader.BERTDataset(test_df, tokenizer, config.BERT_MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False)
    
    # 4. 初始化预训练分类模型
    print(f"[模型加载] 正在加载预训练模型: {config.BERT_MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME, 
        num_labels=2, 
        cache_dir=config.BERT_CACHE_DIR
    ).to(config.DEVICE)
    
    # 5. 优化器与学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.BERT_LR)
    
    total_steps = len(train_loader) * config.BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 6. 训练循环
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print(f"[训练启动] 计划训练 {config.BERT_EPOCHS} 个 Epoch (微调模式)...")
    for epoch in range(config.BERT_EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.BERT_EPOCHS} [Train]"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * input_ids.size(0)
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.BERT_EPOCHS} [Val]"):
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item() * input_ids.size(0)
                _, predicted = logits.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # 记录历史
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
        
        # 保存最佳权重
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # 保存整个模型（包含Tokenizer和权重配置）以便于推理
            model.save_pretrained(config.BERT_WEIGHT_PATH)
            tokenizer.save_pretrained(config.BERT_WEIGHT_PATH)
            print(f" -> 验证集表现提升，模型及Tokenizer已保存至 {config.BERT_WEIGHT_PATH}")
            
    # 7. 在测试集上做最终评估
    print("\n[加载最优模型] 正在加载保存的最佳BERT权重并在测试集上进行最终评估...")
    best_model = AutoModelForSequenceClassification.from_pretrained(config.BERT_WEIGHT_PATH).to(config.DEVICE)
    best_model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"]
            
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
    
    print(f"\n[评估结果] BERT (RoBERTa) 测试集指标:")
    print(f" - Accuracy:  {test_acc:.4f}")
    print(f" - Precision: {test_precision:.4f}")
    print(f" - Recall:    {test_recall:.4f}")
    print(f" - F1-Score:  {test_f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_preds, target_names=["日常/标准 (Label 0)", "网络梗/抽象 (Label 1)"]))
    
    # 保存训练历史
    history_path = os.path.join(config.OUTPUT_DIR, "bert_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
        
    return {
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1
    }

if __name__ == "__main__":
    train_bert_model()
