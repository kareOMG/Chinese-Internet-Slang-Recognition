# Research on Chinese Internet Language Style Recognition based on BERT

[中文说明](./README.md) | English Documentation

---

This project utilizes the `GaryYang123/zh-meme-sft-8k` dataset from Hugging Face to design and implement a classification system for Chinese internet language styles (meme slang, social media roasting, etc.). The project includes three models with varying levels of complexity: **TF-IDF + Logistic Regression**, **BiLSTM + Attention**, and fine-tuned **BERT (RoBERTa)**, along with evaluation and interactive prediction CLI scripts.

---

## 📁 Project Directory Structure

```text
Final_project/
├── data/                       # Dataset directory (created and cached automatically)
│   └── zh_meme_sft_8k_processed.csv
├── output/                     # Training results and visualizations (created automatically)
│   ├── tfidf_lr.pkl            # Logistic Regression model checkpoint
│   ├── bilstm_attention.pth    # BiLSTM model checkpoint
│   ├── bilstm_vocab.json       # BiLSTM vocabulary file
│   ├── bert_classifier/        # Fine-tuned BERT checkpoint & configs
│   ├── model_metrics_comparison.png # Comparison bar chart of 3 models
│   ├── confusion_matrices.png  # Confusion matrices heatmaps
│   ├── training_curves.png     # Loss and accuracy curves
│   └── evaluation_report.txt   # Metrics evaluation text report
├── src/                        # Core source code directory
│   ├── config.py               # Constants, paths, and hyperparameters config
│   ├── data_loader.py          # Data downloading, balancing, splitting & Dataset classes
│   ├── models.py               # BiLSTM + Attention model definition
│   ├── train_baseline.py       # Baseline model training (TF-IDF + LR)
│   ├── train_bilstm.py         # BiLSTM + Attention training script
│   ├── train_bert.py           # BERT (RoBERTa) fine-tuning script
│   ├── evaluate.py             # Performance comparison and plotting
│   └── predict.py              # CLI interactive testing and Attention weights analysis
├── requirements.txt            # Python dependencies
├── main.py                     # One-key pipeline entry script
└── README.md                   # This instruction manual
```

---

## ⚙️ Environment Setup & Configuration

We recommend using Conda to manage your Python virtual environment (Python 3.9 or 3.10 is recommended):

```bash
# 1. Create a new Conda virtual environment
conda create -n slang_rec python=3.10 -y

# 2. Activate the virtual environment
conda activate slang_rec

# 3. Navigate to the project root directory
cd Chinese-Internet-Slang-Recognition

# 4. Install dependencies (including transformers, jieba, matplotlib, seaborn, scikit-learn, etc.)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> [!TIP]
> If you have an NVIDIA GPU and want to accelerate deep learning model training (BiLSTM and BERT), please install the CUDA-supported version of PyTorch according to your local CUDA version from the [PyTorch official website](https://pytorch.org/get-started/locally/). For example:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 Running Guide

You can run the end-to-end pipeline using `main.py` directly, or run individual steps/scripts separately.

### 1. One-key Execution (Recommended)

Run `main.py` in the project root. It will automatically download/clean the dataset, train all 3 models, evaluate their performance, save plots, and launch the interactive CLI prediction mode:

```bash
python main.py
```

### 2. Step-by-Step Running

If you wish to run/debug steps individually, you can use Python's modular run mode `-m` from the project root:

*   **Step 1: Download & Preprocess Data**
    ```bash
    python -m src.data_loader
    ```
*   **Step 2: Train TF-IDF + LR (Baseline)**
    ```bash
    python -m src.train_baseline
    ```
*   **Step 3: Train BiLSTM + Attention (Deep Learning)**
    ```bash
    python -m src.train_bilstm
    ```
*   **Step 4: Fine-tune BERT (RoBERTa)**
    ```bash
    python -m src.train_bert
    ```
*   **Step 5: Run Evaluation & Plotting** (after all models are trained)
    ```bash
    python -m src.evaluate
    ```
*   **Step 6: Launch Interactive Prediction**
    ```bash
    python -m src.predict
    ```

---

## 📊 Evaluation & Outputs

After training, you can find the following evaluation assets in the `output/` directory:
1.  **`evaluation_report.txt`**: Automatically generated metrics table (Accuracy, Precision, Recall, F1-Score) and detailed classification reports.
2.  **`model_metrics_comparison.png`**: Multi-metric bar chart comparing all three models.
3.  **`confusion_matrices.png`**: Confusion matrix heatmap for each of the three models.
4.  **`training_curves.png`**: Training loss and validation accuracy curves for the deep learning models.

---

## 🤝 Dataset Acknowledgement & License

This project utilizes open-source resources, with details declared below:

1. **Dataset Source**: The data preprocessing of this project is based on the [zh-meme-sft-8k](https://huggingface.co/datasets/GaryYang123/zh-meme-sft-8k) dataset open-sourced by **GaryYang123** on the Hugging Face Hub.
2. **License Statement**:
   * The original dataset is licensed under the **MIT License**.
   * This project fully complies with its license terms, providing clear attribution and citations in the code, README, and report.
   * All source code and documentation of this project are open-sourced under the **MIT License**. We welcome anyone to use, modify, or redistribute them.
