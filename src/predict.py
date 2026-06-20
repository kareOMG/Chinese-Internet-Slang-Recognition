import os
import torch
import torch.nn.functional as F
import joblib
import jieba
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import config
from src import data_loader
from src import models

class MultiModelPredictor:
    def __init__(self):
        print("[加载模型] 正在初始化多模型预测器，加载所有训练好的权重...")
        
        # 1. 加载 Baseline
        self.baseline_model = None
        if os.path.exists(config.BASELINE_WEIGHT_PATH):
            self.baseline_data = joblib.load(config.BASELINE_WEIGHT_PATH)
            self.vectorizer = self.baseline_data["vectorizer"]
            self.lr_model = self.baseline_data["model"]
            self.baseline_model = True
            print(" - TF-IDF + Logistic Regression 加载成功。")
        else:
            print(" - [警告] TF-IDF + Logistic Regression 权重文件不存在。")

        # 2. 加载 BiLSTM + Attention
        self.lstm_model = None
        vocab_path = os.path.join(config.OUTPUT_DIR, "bilstm_vocab.json")
        if os.path.exists(config.LSTM_WEIGHT_PATH) and os.path.exists(vocab_path):
            self.vocab = data_loader.Vocab.load(vocab_path)
            self.lstm_model = models.BiLSTMAttention(
                vocab_size=len(self.vocab.word2idx),
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT
            ).to(config.DEVICE)
            self.lstm_model.load_state_dict(torch.load(config.LSTM_WEIGHT_PATH, map_location=config.DEVICE))
            self.lstm_model.eval()
            print(" - BiLSTM + Attention 加载成功。")
        else:
            print(" - [警告] BiLSTM + Attention 权重或词表不存在。")

        # 3. 加载 BERT (RoBERTa)
        self.bert_model = None
        if os.path.exists(config.BERT_WEIGHT_PATH):
            self.bert_tokenizer = AutoTokenizer.from_pretrained(config.BERT_WEIGHT_PATH)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(config.BERT_WEIGHT_PATH).to(config.DEVICE)
            self.bert_model.eval()
            print(" - BERT (RoBERTa) 模型与分词器加载成功。")
        else:
            print(" - [警告] BERT 权重模型目录不存在。")

    def predict(self, text):
        """
        对输入的文本进行风格识别预测
        """
        text = text.strip()
        if not text:
            return
            
        print("\n" + "-"*30 + f" 预测输入: \"{text}\" " + "-"*30)
        
        # 1. 逻辑回归预测
        if self.baseline_model:
            X_vec = self.vectorizer.transform([text])
            probs = self.lr_model.predict_proba(X_vec)[0]
            # label 1 为网络梗风格
            prob_meme = probs[1]
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[TF-IDF + LR]  网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")
        
        # 2. BiLSTM + Attention 预测与注意力权重分析
        if self.lstm_model:
            words = list(jieba.cut(text))
            indices = self.vocab.text_to_indices(text, config.LSTM_MAX_LEN)
            inputs = torch.tensor([indices], dtype=torch.long).to(config.DEVICE)
            
            with torch.no_grad():
                logits, attn_weights = self.lstm_model(inputs)
                probs = F.softmax(logits, dim=1)[0]
                prob_meme = probs[1].item()
                
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[BiLSTM+Attn]  网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")
            
            # 可视化注意力机制所关注的词
            # 取出当前文本长度对应的注意力权重
            attn_scores = attn_weights[0][:len(words)].cpu().numpy()
            if len(attn_scores) > 0:
                # 归一化显示
                attn_scores = attn_scores / (np.sum(attn_scores) + 1e-9)
                print(" -> 词汇注意力权重分析:")
                attn_str_list = []
                for word, score in zip(words, attn_scores):
                    attn_str_list.append(f"{word}({score*100:.1f}%)")
                print("    " + " | ".join(attn_str_list))
                
        # 3. BERT 预测
        if self.bert_model:
            encoding = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=config.BERT_MAX_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(config.DEVICE)
            attention_mask = encoding["attention_mask"].to(config.DEVICE)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)[0]
                prob_meme = probs[1].item()
                
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[BERT (RoB)]   网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")

def interactive_loop():
    predictor = MultiModelPredictor()
    print("\n" + "="*25 + " 中文互联网语言风格识别测试 " + "="*25)
    print("输入 'exit' 或 'q' 退出程序。")
    print("现在可以输入您希望测试的句子，系统将实时输出三种模型的预测结果：\n")

    while True:
        try:
            user_input = input("\n请输入测试文本 >>> ")
            if user_input.strip().lower() in ['exit', 'q']:
                print("\n[退出测试] 测试程序已结束。")
                break
            predictor.predict(user_input)
        except KeyboardInterrupt:
            print("\n[退出测试] 测试程序已结束。")
            break
        except Exception as e:
            print(f"预测出错: {e}")

if __name__ == "__main__":
    interactive_loop()
