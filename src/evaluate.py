import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI模式，避免弹窗报错
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import config
from src import data_loader
from src import models

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_models_and_evaluate():
    print("\n" + "="*20 + " 4. 开始综合评估与对比分析 " + "="*20)
    
    # 1. 准备测试数据
    df = data_loader.download_and_preprocess_data()
    _, _, test_df = data_loader.get_train_val_test_splits(df)
    y_test = test_df["label"].values
    
    results = {}
    
    # ==================== 评估 Model 1: TF-IDF + LR ====================
    print("\n[加载评估] Model 1: TF-IDF + Logistic Regression...")
    if os.path.exists(config.BASELINE_WEIGHT_PATH):
        baseline_data = joblib.load(config.BASELINE_WEIGHT_PATH)
        vectorizer = baseline_data["vectorizer"]
        lr_model = baseline_data["model"]
        
        X_test_baseline = vectorizer.transform(test_df["text"])
        preds_lr = lr_model.predict(X_test_baseline)
        
        acc, prec, rec, f1 = compute_metrics(y_test, preds_lr)
        results["TF-IDF + LR"] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
            "preds": preds_lr
        }
        print(" -> 评估完成。")
    else:
        print(" -> [警告] 未找到 TF-IDF + LR 的权重文件，跳过评估。")

    # ==================== 评估 Model 2: BiLSTM + Attention ====================
    print("\n[加载评估] Model 2: BiLSTM + Attention...")
    vocab_path = os.path.join(config.OUTPUT_DIR, "bilstm_vocab.json")
    if os.path.exists(config.LSTM_WEIGHT_PATH) and os.path.exists(vocab_path):
        vocab = data_loader.Vocab.load(vocab_path)
        lstm_model = models.BiLSTMAttention(
            vocab_size=len(vocab.word2idx),
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        lstm_model.load_state_dict(torch.load(config.LSTM_WEIGHT_PATH, map_location=config.DEVICE))
        lstm_model.eval()
        
        # 转换为 Dataset & DataLoader
        test_dataset = data_loader.BiLSTMDataset(test_df, vocab, config.LSTM_MAX_LEN)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
        
        preds_lstm = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(config.DEVICE)
                logits, _ = lstm_model(inputs)
                _, predicted = logits.max(1)
                preds_lstm.extend(predicted.cpu().numpy())
                
        preds_lstm = np.array(preds_lstm)
        acc, prec, rec, f1 = compute_metrics(y_test, preds_lstm)
        results["BiLSTM + Attention"] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
            "preds": preds_lstm
        }
        print(" -> 评估完成。")
    else:
        print(" -> [警告] 未找到 BiLSTM 权重或词表文件，跳过评估。")

    # ==================== 评估 Model 3: BERT ====================
    print("\n[加载评估] Model 3: BERT (RoBERTa)...")
    if os.path.exists(config.BERT_WEIGHT_PATH):
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_WEIGHT_PATH)
        bert_model = AutoModelForSequenceClassification.from_pretrained(config.BERT_WEIGHT_PATH).to(config.DEVICE)
        bert_model.eval()
        
        test_dataset = data_loader.BERTDataset(test_df, tokenizer, config.BERT_MAX_LEN)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False)
        
        preds_bert = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = outputs.logits.max(1)
                preds_bert.extend(predicted.cpu().numpy())
                
        preds_bert = np.array(preds_bert)
        acc, prec, rec, f1 = compute_metrics(y_test, preds_bert)
        results["BERT (RoBERTa)"] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
            "preds": preds_bert
        }
        print(" -> 评估完成。")
    else:
        print(" -> [警告] 未找到 BERT 的权重目录，跳过评估。")

    if not results:
        print("[错误] 未找到任何有效的模型权重进行评估，请先运行训练脚本。")
        return

    # ==================== 性能指标对比表格 ====================
    metrics_data = []
    for model_name, metrics in results.items():
        metrics_data.append({
            "模型名称": model_name,
            "Accuracy (准确率)": f"{metrics['Accuracy']:.4f}",
            "Precision (精确率)": f"{metrics['Precision']:.4f}",
            "Recall (召回率)": f"{metrics['Recall']:.4f}",
            "F1-Score (F1值)": f"{metrics['F1-Score']:.4f}"
        })
    df_metrics = pd.DataFrame(metrics_data)
    print("\n" + "="*20 + " 各模型测试集对比指标 " + "="*20)
    print(df_metrics.to_string(index=False))
    
    # 保存指标表格到本地
    report_text_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.txt")
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("中文互联网语言风格识别研究 - 模型对比报告\n")
        f.write("="*50 + "\n\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\n详细分类报告:\n")
        for model_name, metrics in results.items():
            f.write(f"\n[{model_name}] 分类报告:\n")
            f.write(classification_report(y_test, metrics["preds"], target_names=["日常/标准 (Label 0)", "网络梗/抽象 (Label 1)"]))
    print(f"\n[文本报告] 详细文本报告已保存至 {report_text_path}")

    # ==================== 绘图 1: 指标对比柱状图 ====================
    plot_metrics_comparison(results)

    # ==================== 绘图 2: 混淆矩阵热力图 ====================
    plot_confusion_matrices(results, y_test)

    # ==================== 绘图 3: 训练 Loss & Acc 曲线 ====================
    plot_training_curves()

    print(f"\n[可视化图表] 所有对比图表已保存至 {config.OUTPUT_DIR} 目录下。")

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return acc, prec, rec, f1

def plot_metrics_comparison(results):
    models_names = list(results.keys())
    accuracies = [results[m]["Accuracy"] for m in models_names]
    precisions = [results[m]["Precision"] for m in models_names]
    recalls = [results[m]["Recall"] for m in models_names]
    f1_scores = [results[m]["F1-Score"] for m in models_names]
    
    x = np.arange(len(models_names))
    width = 0.18
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy (准确率)', color='#3498db')
    plt.bar(x - 0.5*width, precisions, width, label='Precision (精确率)', color='#2ecc71')
    plt.bar(x + 0.5*width, recalls, width, label='Recall (召回率)', color='#e67e22')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score (F1值)', color='#9b59b6')
    
    plt.ylabel('Score (得分)', fontsize=12)
    plt.title('三大模型各项评估指标综合对比图', fontsize=14, pad=15)
    plt.xticks(x, models_names, fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 在柱子上添加具体数值
    for i in range(len(models_names)):
        plt.text(i - 1.5*width, accuracies[i] + 0.005, f"{accuracies[i]:.3f}", ha='center', va='bottom', fontsize=8)
        plt.text(i - 0.5*width, precisions[i] + 0.005, f"{precisions[i]:.3f}", ha='center', va='bottom', fontsize=8)
        plt.text(i + 0.5*width, recalls[i] + 0.005, f"{recalls[i]:.3f}", ha='center', va='bottom', fontsize=8)
        plt.text(i + 1.5*width, f1_scores[i] + 0.005, f"{f1_scores[i]:.3f}", ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "model_metrics_comparison.png"), dpi=300)
    plt.close()

def plot_confusion_matrices(results, y_true):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
        
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = confusion_matrix(y_true, metrics["preds"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=["日常", "网络梗"], yticklabels=["日常", "网络梗"],
                    annot_kws={"size": 14})
        axes[idx].set_title(f"{model_name}\n混淆矩阵", fontsize=13)
        axes[idx].set_xlabel("预测标签", fontsize=11)
        axes[idx].set_ylabel("真实标签", fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()

def plot_training_curves():
    bilstm_hist_path = os.path.join(config.OUTPUT_DIR, "bilstm_history.json")
    bert_hist_path = os.path.join(config.OUTPUT_DIR, "bert_history.json")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 绘制 Loss 曲线
    has_plot = False
    if os.path.exists(bilstm_hist_path):
        with open(bilstm_hist_path, "r") as f:
            b_hist = json.load(f)
        ax1.plot(b_hist["train_loss"], label="BiLSTM 训练 Loss", linestyle="-", color="#e74c3c")
        ax1.plot(b_hist["val_loss"], label="BiLSTM 验证 Loss", linestyle="--", color="#c0392b")
        
        ax2.plot(b_hist["val_acc"], label="BiLSTM 验证 Accuracy", linestyle="-", color="#2ecc71")
        has_plot = True
        
    if os.path.exists(bert_hist_path):
        with open(bert_hist_path, "r") as f:
            bert_hist = json.load(f)
        ax1.plot(bert_hist["train_loss"], label="BERT 训练 Loss", linestyle="-", color="#3498db")
        ax1.plot(bert_hist["val_loss"], label="BERT 验证 Loss", linestyle="--", color="#2980b9")
        
        ax2.plot(bert_hist["val_acc"], label="BERT 验证 Accuracy", linestyle="-", color="#9b59b6")
        has_plot = True
        
    if has_plot:
        ax1.set_title("深度学习模型训练与验证损失曲线 (Loss Curves)", fontsize=12)
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=9)
        
        ax2.set_title("深度学习模型验证集准确率曲线 (Val Accuracy)", fontsize=12)
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "training_curves.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    load_models_and_evaluate()
