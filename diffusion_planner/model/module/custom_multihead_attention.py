import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomMultiheadAttention(nn.Module):
    """
    一个与 nn.MultiheadAttention 参数命名兼容的、ONNX友好的多头注意力实现。
    参数结构保持不变以兼容已有模型的参数加载。
    """
    def __init__(self, dim, heads=6, dropout=0.1, batch_first=True):
        super().__init__()
        assert batch_first, "CustomMultiheadAttention only supports batch_first=True"
        assert dim % heads == 0, "Embedding dimension must be divisible by number of heads"

        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # 保持原有参数结构，确保兼容性
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        """使用与官方相同的初始化方法"""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        B, T_q, C = query.shape
        _, T_k, _ = key.shape
        _, T_v, _ = value.shape
        
        # 确保K和V的序列长度一致
        assert T_k == T_v, "Key and Value must have the same sequence length"

        # 判断是否为自注意力并且所有输入相同
        is_self_attention = (torch.equal(query, key) and torch.equal(key, value) and T_q == T_k)
        
        if is_self_attention:
            # 自注意力：一次性计算QKV，这是官方实现的优化路径
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # 非自注意力：分别计算Q, K, V
            # 分离权重和偏置
            q_weight = self.in_proj_weight[:self.dim]
            k_weight = self.in_proj_weight[self.dim:2*self.dim]  
            v_weight = self.in_proj_weight[2*self.dim:]
            
            q_bias = self.in_proj_bias[:self.dim]
            k_bias = self.in_proj_bias[self.dim:2*self.dim]
            v_bias = self.in_proj_bias[2*self.dim:]
            
            q = F.linear(query, q_weight, q_bias)
            k = F.linear(key, k_weight, k_bias)
            v = F.linear(value, v_weight, v_bias)

        # Reshape为多头格式: (B, T, D) -> (B, T, H, D_h) -> (B, H, T, D_h)
        q = q.reshape(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T_v, self.num_heads, self.head_dim).transpose(1, 2)

        # 使用scaled dot-product attention的标准实现
        # 不在这里预先缩放q，而是在attention计算中处理
        attn_output = self._scaled_dot_product_attention(
            q, k, v, key_padding_mask, self.attn_dropout if self.training else None
        )

        # Reshape回原始格式: (B, H, T_q, D_h) -> (B, T_q, H, D_h) -> (B, T_q, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, T_q, C)

        # 最终线性投影
        x = self.out_proj(attn_output)
        x = self.proj_dropout(x)
        
        return x
    
    def _scaled_dot_product_attention(self, q, k, v, key_padding_mask=None, dropout_fn=None):
        """
        实现标准的scaled dot-product attention，严格按照官方逻辑
        """
        B, H, T_q, D_h = q.shape
        T_k = k.shape[2]
        
        # 计算attention scores，这里才应用缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_h)
        
        # 应用key_padding_mask
        if key_padding_mask is not None:
            # 确保mask是正确的bool类型
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            
            # key_padding_mask的形状应该是(B, T_k)
            assert key_padding_mask.shape == (B, T_k), f"Expected key_padding_mask shape ({B}, {T_k}), got {key_padding_mask.shape}"
            
            # 扩展mask的维度以匹配scores: (B, T_k) -> (B, 1, 1, T_k)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            
            # 在PyTorch的官方实现中，key_padding_mask的True表示忽略（padding），False表示保留（有效）
            # 所以我们需要将True的位置设为-inf
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算attention权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 处理可能的NaN（当整行都是-inf时softmax会产生NaN）
        if key_padding_mask is not None:
            # 如果某个query位置对应的所有key都被mask掉，那么attention权重会是NaN
            # 我们需要将这些NaN替换为0
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # 应用dropout
        if dropout_fn is not None:
            attn_weights = dropout_fn(attn_weights)
        
        # 应用attention权重到value
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output