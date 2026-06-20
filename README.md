# 基于 BERT 的中文互联网语言风格识别研究 (期末项目)

本项目使用 Hugging Face 上的 `GaryYang123/zh-meme-sft-8k` 数据集，设计并实现了基于 BERT 的中文互联网语言风格（抽象梗、社交吐槽语体）识别系统。本项目包含三种不同复杂度的模型：**TF-IDF + 逻辑回归**、**BiLSTM + Attention**、以及 **BERT (RoBERTa)** 微调模型，并提供了对比与交互预测环境。

---

## 📁 项目目录结构

```text
Final_project/
├── data/                       # 数据集目录 (自动创建并缓存)
│   └── zh_meme_sft_8k_processed.csv
├── output/                     # 训练结果与可视化图表 (自动创建)
│   ├── tfidf_lr.pkl            # 逻辑回归模型权重
│   ├── bilstm_attention.pth    # BiLSTM 模型权重
│   ├── bilstm_vocab.json       # BiLSTM 词表
│   ├── bert_classifier/        # BERT 微调权重及配置文件
│   ├── model_metrics_comparison.png # 三模型性能对比图
│   ├── confusion_matrices.png  # 混淆矩阵热力图
│   ├── training_curves.png     # 训练损失与精度曲线
│   └── evaluation_report.txt   # 文本格式的评估报告
├── src/                        # 核心代码目录
│   ├── config.py               # 项目路径与模型超参数配置
│   ├── data_loader.py          # 数据清洗、平衡、划分与 DataLoader 封装
│   ├── models.py               # BiLSTM + Attention 模型定义
│   ├── train_baseline.py       # 训练基线模型 (TF-IDF + LR)
│   ├── train_bilstm.py         # 训练经典深度学习模型 (BiLSTM + Attention)
│   ├── train_bert.py           # 微调 BERT 模型 (RoBERTa-wwm-ext)
│   ├── evaluate.py             # 综合评估、对比分析与图表生成
│   └── predict.py              # 命令行交互预测与注意力权重分析
├── requirements.txt            # 项目 Python 依赖
├── main.py                     # 项目一键运行总入口
└── README.md                   # 本说明文件
```

---

## ⚙️ 环境安装与配置

推荐使用 Conda 创建并管理 Python 虚拟环境（建议使用 Python 3.9 或 3.10 版本）：

```bash
# 1. 创建全新的 Conda 虚拟环境
conda create -n slang_rec python=3.10 -y

# 2. 激活虚拟环境
conda activate slang_rec

# 3. 切换到项目根目录
cd Chinese-Internet-Slang-Recognition

# 4. 安装依赖库 (包含 transformers, jieba, matplotlib, seaborn, scikit-learn 等)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> [!TIP]
> 如果您拥有 NVIDIA 显卡并希望使用 GPU 加速模型训练（BiLSTM 与 BERT），请根据您的 CUDA 版本，提前前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装支持 CUDA 的 PyTorch 版本。例如：
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 运行指南

您可以通过项目根目录下的 `main.py` 运行完整流程，也可以单独执行各个步骤的模型训练和评估。

### 1. 一键运行完整实验流程 (推荐)

在项目根目录下直接运行 `main.py`，它将自动执行：下载数据 ➡️ 清洗平衡 ➡️ 训练 3 款模型 ➡️ 评估对比并保存图表 ➡️ 进入交互预测模式：

```bash
python main.py
```

### 2. 分步单独运行

如果您希望分步进行调试或训练，可在项目根目录下通过 `-m` 模块化运行方式执行各个脚本：

*   **步骤 1：仅下载并预处理数据**
    ```bash
    python -m src.data_loader
    ```
*   **步骤 2：单独训练 TF-IDF + LR (基线模型)**
    ```bash
    python -m src.train_baseline
    ```
*   **步骤 3：单独训练 BiLSTM + Attention (经典深度学习模型)**
    ```bash
    python -m src.train_bilstm
    ```
*   **步骤 4：单独训练 BERT (RoBERTa 预训练微调)**
    ```bash
    python -m src.train_bert
    ```
*   **步骤 5：单独进行对比评估与图表生成** (在三个模型训练完成后运行)
    ```bash
    python -m src.evaluate
    ```
*   **步骤 6：单独启动交互式测试与注意力权重分析**
    ```bash
    python -m src.predict
    ```

---

## 📊 评估指标与结果产出

模型训练完成后，您可以在 `output/` 文件夹中找到以下生成的期末报告支撑材料：
1.  **`evaluation_report.txt`**：自动生成包含准确率(Accuracy)、精确率(Precision)、召回率(Recall)和 F1-Score 的详细指标对比表格。
2.  **`model_metrics_comparison.png`**：三款模型的评估指标对比柱状图。
3.  **`confusion_matrices.png`**：三款模型的预测混淆矩阵。
4.  **`training_curves.png`**：深度学习模型训练过程中的 Loss 下降和 Accuracy 上升曲线图。

---

## 🤝 数据来源致谢与开源声明

本项目在开发过程中合理使用并遵守了外部开源成果，相关声明如下：

1. **数据集来源**：本项目的数据预处理基于 Hugging Face 社区中由 **GaryYang123** 开源的 [zh-meme-sft-8k](https://huggingface.co/datasets/GaryYang123/zh-meme-sft-8k) 数据集。
2. **许可协议说明**：
   * 该数据集原始许可协议为 **MIT License**。
   * 本项目完全遵循其开源协议，在代码、README 及报告中对原作者进行了明确归属与引用。
   * 本项目自身的全部源代码及技术文档同样采用 **MIT License** 开源，允许任何个人或团队自由学习、复制、修改及再分发。

感谢原作者对中文互联网梗文化研究做出的开源贡献！
