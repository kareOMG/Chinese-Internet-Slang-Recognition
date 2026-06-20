import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    点积自注意力机制 (Bahdanau-style Attention)
    用于在 LSTM 的所有隐藏状态中自动寻找对风格识别最关键的词汇。
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs shape: [batch_size, seq_len, hidden_dim]
        # projection shape: [batch_size, seq_len, 1]
        energy = self.projection(lstm_outputs)
        
        # weights shape: [batch_size, seq_len, 1]
        weights = F.softmax(energy, dim=1)
        
        # outputs shape: [batch_size, hidden_dim]
        outputs = torch.sum(lstm_outputs * weights, dim=1)
        return outputs, weights.squeeze(-1)


class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Attention 文本分类模型
    双向 LSTM 能够捕获上下文语义，注意力机制聚焦核心梗词汇，最后通过线性层进行二分类。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 注意力层 (输入维度为 hidden_dim * 2 因为是双向 LSTM)
        self.attention = Attention(hidden_dim * 2)
        
        # 丢弃层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 分类线性层 (输出维度为 2, 表示二分类的两个类别的 logits)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # embed shape: [batch_size, seq_len, embedding_dim]
        embed = self.embedding(x)
        
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(embed)
        
        # context shape: [batch_size, hidden_dim * 2]
        context, attn_weights = self.attention(lstm_out)
        
        # dropout
        context = self.dropout(context)
        
        # logits shape: [batch_size, 2]
        logits = self.fc(context)
        
        return logits, attn_weights
