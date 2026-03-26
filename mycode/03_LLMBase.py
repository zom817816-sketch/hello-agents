import collections 

# 示例语料库
corpus = "ai agent likes ai agent works" 
tokens = corpus.split() 
total_tokens = len(tokens) 

# 计算P("ai")
count_ai = tokens.count("ai")
p_ai = count_ai / total_tokens
print(f"P(ai): = {p_ai:.3f}") 

# 计算P("agent"|"ai") 
bigrams = zip(tokens, tokens[1:]) 
bigram_count = collections.Counter(bigrams) 
count_ai_agent = bigram_count[("ai", "agent")] 
p_agent_given_ai = count_ai_agent / count_ai 
print(f"P(angent|ai) = {p_agent_given_ai:.3f}")

# 计算P("likes"|"agent") 
count_agent_likes = bigram_count[("agent", "likes")] 
count_agent = tokens.count("agent") 
p_likes_given_agent = count_agent_likes / count_agent
print(f"p(likes|agent) = {p_likes_given_agent:.3f}") 

# 最后将概率连乘
p_sentence = p_ai * p_agent_given_ai * p_likes_given_agent 
print(f"P(ai agent likes) = {p_sentence:.3f}")

# N-gram的致命缺陷:
# 数据稀疏性:如果一个词从未在语料库中出现过则概率估计为0
# 泛化性差:无法理解词与词之间的语义相似性
# 无法理解语义关系:纯粹基于统计共线
# 存储开销大:需要存储所有n_gram的计数,内存需求随词汇量增长指数增长

import numpy as np 

# 已学习的嵌入向量 
embeddings = {
    "king": np.array([0.9, 0.8]), 
    "queen": np.array([0.9, 0.2]), 
    "man": np.array([0.7, 0.9]), 
    "woman": np.array([0.7, 0.3])
}

def cos_similarity(vec1, vec2): 
    dot_product = np.dot(vec1, vec2) 
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2) # 计算L2范数
    return dot_product / norm_product 

# king - man + woman 
res = embeddings["king"] - embeddings["man"] + embeddings["woman"] 

# 计算余弦相似度
sim = cos_similarity(embeddings["queen"], res) 

print(f"king - man + woman 的结果向量: {res}")
print(f"该结果与 'queen' 的相似度: {sim:.4f}")

# Transformer实现
import torch 
import torch.nn as nn 
import math 

# 位置编码模块 
class PosEncoding(nn.Module): 
    """
    位置编码模块
    """ 
    def __init__(self, d_model, dropout, max_len): 
        super().__init__() 
        self.dropout = dropout 

        # 创建一个足够长的位置编码矩阵 
        pos = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model) # shape: max_len, d_model
        pe[:, 0::2] = torch.sin(pos * div_term) # 偶数用sin
        pe[:, 1::2] = torch.cos(pos * div_term) # 奇数用cos 

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x) 

# 多头注意力机制
class MultiHeadAttention(nn.Module): 
    """
    多头注意力机制
    """ 
    def __init__(self, d_model, num_heads): 
        super().__init__() 
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除" 
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads  

        self.wq = nn.Linear(d_model, d_model) 
        self.wk = nn.Linear(d_model, d_model) 
        self.wv = nn.Linear(d_model, d_model) 
        self.wo = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None): 
        batch_size, seq_len, d_model = Q.shape # shape是属性而不是方法,size()是方法

        Q = self.wq(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 
        V = self.wv(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: 
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) 
        attn_probs = self.attn_dropout(torch.softmax(attn_scores, dim=-1)) # 注意softmax需要dim参数
        attn_output = torch.matmul(attn_probs, V) # shape:batch_size, num_heads, seq_len, d_k 

        attn_output_combined = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model) 

        output = self.wo(attn_output_combined)

        return output
         
# 前馈层
class MLP(nn.Module): 
    """ 
    位置前馈网络
    """ 
    def __init__(self,d_model, d_ff, dropout): 
        super().__init__() 
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.linear2 = nn.Linear(d_ff, d_model) 
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x): 
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

# RMSNorm 
class RMSNorm(nn.Module): 
    def __init__(self, dim, eps=1e-6): 
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) 
        self.eps = eps 

    def _norm(self, x): 
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    def forward(self, x): 
        output = self._norm(x.float()).type_as(x) 
        return self.weight * output 

# LayerNorm 
class LayerNorm(nn.Module): 
    def __init__(self, dim, eps=1e-6): 
        super().__init__() 
        self.dim = dim 
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(dim)) 
        self.beta = nn.Parameter(torch.zeros(dim)) 

    def forward(self, x: torch.Tensor): 
        mean = torch.mean(x, dim=-1, keepdim=True) 
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False) 
        x_norm = (x - mean) * torch.rsqrt(var + self.eps) 
        return self.gamma * x_norm + beta 

# 编码器核心层
class EncoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout): 
        super().__init__() 
        self.self_attn = MultiHeadAttention() 
        self.mlp = MLP() 
        self.attn_norm = RMSNorm(d_model) 
        self.mlp_norm = RMSNorm(d_model) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask): 
        x_norm = self.attn_norm(x)
        # 残差连接:提供跳跃路径,防止梯度清零;允许模型训练数百甚至上千层;缓解退化问题,加速收敛;网络可以选择性的使用层
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask) + x 
        output_norm = self.mlp_norm(attn_output) 
        mlp_output = self.dropout(self.mlp(x_norm)) + x 
        return mlp_output

# 解码器核心层 
class DecoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout): 
        super().__init__() 
        self.self_attn = MultiHeadAttention() 
        self.cross_attn = MultiHeadAttention() 
        self.mlp = MLP() 
        self.norm1 = RMSNorm(d_model) 
        self.norm2 = RMSNorm(d_model) 
        self.norm3 = RMSNorm(d_model) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        x_norm = self.norm1(x) 
        self_attn_ouput = self.self_attn(x_norm, x_norm, x_norm, tgt_mask) + x 
        output_norm = self.norm2(self_attn_ouput)
        cross_attn_output = self.cross_attn(output_norm, encoder_output, encoder_output, src_mask) + x 
        output_norm = self.norm3(self_attn_ouput) 
        output = self.dropout(self.mlp(output_norm)) + x 
        return output 
