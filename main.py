import os
# 设置 Hugging Face 镜像源以确保国内网络下载模型成功
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from src.train_baseline import train_baseline_model
from src.train_bilstm import train_bilstm_model
from src.train_bert import train_bert_model
from src.evaluate import load_models_and_evaluate
from src.predict import interactive_loop
from src import data_loader

def print_banner():
    banner = """
===================================================================
*        基于 BERT 的中文互联网语言风格识别研究 (期末项目)         *
*      识别中文网络流行语、抽象梗与标准日常语体 (多模型对比)        *
===================================================================
    """
    print(banner)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="中文互联网语言风格识别项目入口")
    parser.add_argument("--download_data", action="store_true", help="仅下载并预处理数据集")
    parser.add_argument("--train_baseline", action="store_true", help="仅训练 TF-IDF + Logistic Regression")
    parser.add_argument("--train_bilstm", action="store_true", help="仅训练 BiLSTM + Attention")
    parser.add_argument("--train_bert", action="store_true", help="仅训练/微调 BERT")
    parser.add_argument("--evaluate", action="store_true", help="仅进行模型综合评估与可视化")
    parser.add_argument("--predict", action="store_true", help="进入命令行交互预测模式")
    parser.add_argument("--all", action="store_true", help="执行完整流程: 下载、训练全部模型、评估并进入预测")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，默认执行 --all
    no_args = not (args.download_data or args.train_baseline or args.train_bilstm or 
                   args.train_bert or args.evaluate or args.predict)
    
    if args.all or no_args:
        print("[主控制流] 开始执行完整实验流水线...")
        # 1. 确保数据存在
        data_loader.download_and_preprocess_data()
        
        # 2. 依次训练模型
        train_baseline_model()
        train_bilstm_model()
        train_bert_model()
        
        # 3. 评估模型
        load_models_and_evaluate()
        
        # 4. 预测模式
        interactive_loop()
        
    else:
        if args.download_data:
            data_loader.download_and_preprocess_data()
        if args.train_baseline:
            train_baseline_model()
        if args.train_bilstm:
            train_bilstm_model()
        if args.train_bert:
            train_bert_model()
        if args.evaluate:
            load_models_and_evaluate()
        if args.predict:
            interactive_loop()

if __name__ == "__main__":
    main()
