import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.to(attn_mask.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0
        self.x=None

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x,x_m, task = 7, register_hook=False, get_feat=False, get_cur_feat=False, key_padding_mask=None, attn_mask=None):

        if get_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)  # Assuming mask is of shape (batch_size, seq_len)
            self.matrix = (self.matrix * self.n_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_matrix + x.shape[0] * x.shape[1])
            self.n_matrix += x.shape[0] * x.shape[1]
        if get_cur_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)    # Assuming mask is of shape (batch_size, seq_len)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_cur_matrix + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0] * x.shape[1]

        self.x=x_m
        bsz=x.shape[0]
        src_len=x.shape[1]
        tgt_len=src_len
        embed_dim=x.shape[2]
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (x.shape[0] , x.shape[1]), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(x.shape[0], 1, 1, x.shape[1]).\
                expand(-1, self.num_heads, -1, -1).reshape(x.shape[0] * self.num_heads, 1, x.shape[1])
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
    
        if task > -0.5:
            
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        k = k.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        v = v.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]

        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        if attn_mask is not None:
            attn_mask = attn_mask.to(q_scaled.dtype) 
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)).transpose(0, 1)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        # print(x.shape)
        return x


    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v

class Attention_LoRA_our(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0
        self.x=None
        # self.gate_hidden = 128
        # self.gate = nn.Sequential(
        #     nn.Linear(dim, self.gate_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.gate_hidden, n_tasks)  # 输出 [bs, n_tasks] 的 logits
        # )
        self.gate = nn.Linear(dim, n_tasks, bias=False)

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x,x_m, task = 7, register_hook=False, get_feat=False, get_cur_feat=False, key_padding_mask=None, attn_mask=None):

        if get_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)  # Assuming mask is of shape (batch_size, seq_len)
            self.matrix = (self.matrix * self.n_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_matrix + x.shape[0] * x.shape[1])
            self.n_matrix += x.shape[0] * x.shape[1]
        if get_cur_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)    # Assuming mask is of shape (batch_size, seq_len)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_cur_matrix + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0] * x.shape[1]

        self.x=x_m
        bsz=x.shape[0]
        src_len=x.shape[1]
        tgt_len=src_len
        embed_dim=x.shape[2]
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (x.shape[0] , x.shape[1]), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(x.shape[0], 1, 1, x.shape[1]).\
                expand(-1, self.num_heads, -1, -1).reshape(x.shape[0] * self.num_heads, 1, x.shape[1])
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        pooled = x.mean(dim=1)                  # [B, C]
        gate_logits = self.gate(pooled)         # [B, n_tasks]
        gate_logits = gate_logits[:, :task+1]
        a = torch.softmax(gate_logits, dim=-1)*(task+1)  # [B, task+1]
        # print("gain:",a)
        delta_k = 0.0
        delta_v = 0.0
        for t in range(task + 1):
            Wk_t = torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight)
            Wv_t = torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight)

            dk_t = F.linear(x, Wk_t)   # [B,N,C]
            dv_t = F.linear(x, Wv_t)   # [B,N,C]

            coef = a[:, t].view(B, 1, 1)
            delta_k = delta_k + coef * dk_t
            delta_v = delta_v + coef * dv_t

        k = k + delta_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # if task > -0.5:
        #     # x: [B,N,C]
        #     pooled = x.mean(dim=1)               # [B,C]  (也可用 x[:,0] CLS)
        #     gate_logits = self.gate(pooled)      # [B, n_tasks]

        #     # 只用历史 + 当前分支
        #     gate_logits = gate_logits[:, :task+1]        # [B, task+1]
        #     a = torch.softmax(gate_logits, dim=-1)       # [B, task+1]

        #     # 计算每个分支的 delta_k, delta_v，再按 a 加权求和
        #     delta_k = 0.0
        #     delta_v = 0.0
        #     for t in range(task + 1):
        #         Wk_t = self.lora_B_k[t].weight @ self.lora_A_k[t].weight   # [dim, dim]
        #         Wv_t = self.lora_B_v[t].weight @ self.lora_A_v[t].weight   # [dim, dim]

        #         dk_t = F.linear(x, Wk_t)   # [B,N,C]
        #         dv_t = F.linear(x, Wv_t)   # [B,N,C]

        #         coef = a[:, t].view(B, 1, 1)   # [B,1,1] broadcast
        #         delta_k = delta_k + coef * dk_t
        #         delta_v = delta_v + coef * dv_t

        #     k = k + delta_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     v = v + delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)       
        # if task > -0.5:
            
        #     weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
        #     weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
        #     k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # weight_k = self.lora_B_k[task].weight @ self.lora_A_k[task].weight
            # weight_v = self.lora_B_v[task].weight @ self.lora_A_v[task].weight
            # k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        k = k.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        v = v.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]

        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        if attn_mask is not None:
            attn_mask = attn_mask.to(q_scaled.dtype) 
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)).transpose(0, 1)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        # print(x.shape)
        return x


    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v


class Attention_LoRA_alone(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r
        n_tasks=1
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0
        self.x=None
        # self.gate_hidden = 128
        # self.gate = nn.Sequential(
        #     nn.Linear(dim, self.gate_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.gate_hidden, n_tasks)  # 输出 [bs, n_tasks] 的 logits
        # )
        # self.gate = nn.Linear(dim, n_tasks, bias=False)

        # self.gate_dropout = nn.Dropout(0.1) 
    
    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x,x_m, task = 7, register_hook=False, get_feat=False, get_cur_feat=False, key_padding_mask=None, attn_mask=None):

        if get_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)  # Assuming mask is of shape (batch_size, seq_len)
            self.matrix = (self.matrix * self.n_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_matrix + x.shape[0] * x.shape[1])
            self.n_matrix += x.shape[0] * x.shape[1]
        if get_cur_feat:
            x_masked = x * (~key_padding_mask).unsqueeze(-1)    # Assuming mask is of shape (batch_size, seq_len)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x_masked.detach().permute(0, 2, 1), x_masked.detach()).sum(dim=0).cpu()) / (self.n_cur_matrix + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0] * x.shape[1]

        self.x=x_m
        bsz=x.shape[0]
        src_len=x.shape[1]
        tgt_len=src_len
        embed_dim=x.shape[2]
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (x.shape[0] , x.shape[1]), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(x.shape[0], 1, 1, x.shape[1]).\
                expand(-1, self.num_heads, -1, -1).reshape(x.shape[0] * self.num_heads, 1, x.shape[1])
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # pooled = x.mean(dim=1)                  # [B, C]
        # gate_logits = self.gate(pooled)         # [B, n_tasks]
        # gate_logits = gate_logits[:, :task+1]
        # a = torch.softmax(gate_logits, dim=-1)  # [B, task+1]
        # # print("gain:",a)
        # # a = self.gate_dropout(a)
        # delta_k = 0.0
        # delta_v = 0.0
        # for t in range(task + 1):
        #     Wk_t = torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight)
        #     Wv_t = torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight)

        #     dk_t = F.linear(x, Wk_t)   # [B,N,C]
        #     dv_t = F.linear(x, Wv_t)   # [B,N,C]

        #     coef = a[:, t].view(B, 1, 1)
        #     delta_k = delta_k + coef * dk_t
        #     delta_v = delta_v + coef * dv_t

        # k = k + delta_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = v + delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # if task > -0.5:
        #     # x: [B,N,C]
        #     pooled = x.mean(dim=1)               # [B,C]  (也可用 x[:,0] CLS)
        #     gate_logits = self.gate(pooled)      # [B, n_tasks]

        #     # 只用历史 + 当前分支
        #     gate_logits = gate_logits[:, :task+1]        # [B, task+1]
        #     a = torch.softmax(gate_logits, dim=-1)       # [B, task+1]

        #     # 计算每个分支的 delta_k, delta_v，再按 a 加权求和
        #     delta_k = 0.0
        #     delta_v = 0.0
        #     for t in range(task + 1):
        #         Wk_t = self.lora_B_k[t].weight @ self.lora_A_k[t].weight   # [dim, dim]
        #         Wv_t = self.lora_B_v[t].weight @ self.lora_A_v[t].weight   # [dim, dim]

        #         dk_t = F.linear(x, Wk_t)   # [B,N,C]
        #         dv_t = F.linear(x, Wv_t)   # [B,N,C]

        #         coef = a[:, t].view(B, 1, 1)   # [B,1,1] broadcast
        #         delta_k = delta_k + coef * dk_t
        #         delta_v = delta_v + coef * dv_t

        #     k = k + delta_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     v = v + delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        task=1      
        # if task > -0.5:
            
        # 假设你只需要使用一个 LoRA 参数
        weight_k = torch.mm(self.lora_B_k[0].weight, self.lora_A_k[0].weight)  # 使用单个 LoRA 参数
        weight_v = torch.mm(self.lora_B_v[0].weight, self.lora_A_v[0].weight)  # 使用单个 LoRA 参数

        # 对于 k 和 v 使用更新后的权重
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # weight_k = self.lora_B_k[task].weight @ self.lora_A_k[task].weight
            # weight_v = self.lora_B_v[task].weight @ self.lora_A_v[task].weight
            # k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        k = k.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]
        v = v.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [bsz * num_heads, N, head_dim]

        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        if attn_mask is not None:
            attn_mask = attn_mask.to(q_scaled.dtype) 
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)).transpose(0, 1)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        # print(x.shape)
        return x


    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v
        
class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        return x * self.param

class Attention_SDLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rank = r

        # Initialize LoRA components for each task
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.gate = nn.Linear(dim, n_tasks, bias=False)
        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.scaling_factor = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.scaling_factor_prev = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.8]))) for _ in range(n_tasks)])

    def init_param(self):
        # Initialize parameters for LoRA
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x,x_m, task = 7, register_hook=False, get_feat=False, get_cur_feat=False, key_padding_mask=None, attn_mask=None):
        B, N, C = x.shape

        # Compute qkv from input
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Query, Key, Value
        src_len=x.shape[1]
        tgt_len=src_len
        bsz=x.shape[0]
        embed_dim=x.shape[2]
        # Compute the gate values
        pooled = x.mean(dim=1)  # [B, C]
        gate_logits = self.gate(pooled)  # [B, n_tasks]
        gate_logits = gate_logits[:, :task + 1]
        a = torch.softmax(gate_logits, dim=-1)  # [B, task+1]

        # Initialize deltas for key and value
        delta_k = 0.0
        delta_v = 0.0

        # Loop through each task and compute the corresponding update (delta)
        for t in range(task + 1):
            # Get the LoRA matrices for this task
            Wk_t = torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight)
            Wv_t = torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight)

            # Compute the delta for key and value
            dk_t = F.linear(x, Wk_t)  # [B, N, C]
            dv_t = F.linear(x, Wv_t)  # [B, N, C]

            # Get the scaling factor for this task
            coef = a[:, t].view(B, 1, 1)

            # Apply scaling factors for both previous and current tasks
            # Using scaling_factor_prev for previous tasks
            if t < task:
                delta_k += coef * self.scaling_factor_prev[t].param * dk_t
                delta_v += coef * self.scaling_factor_prev[t].param * dv_t
            # Using scaling_factor for the current task
            elif t == task:
                delta_k += coef * self.scaling_factor[0].param * dk_t
                delta_v += coef * self.scaling_factor[0].param * dv_t

        # Add the deltas to the key and value
        k = k + delta_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Reshape q, k, v for the attention calculation
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [B * num_heads, N, head_dim]
        k = k.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [B * num_heads, N, head_dim]
        v = v.reshape(B * self.num_heads, N, C // self.num_heads)  # Shape: [B * num_heads, N, head_dim]

        # Scale the query
        q_scaled = q * math.sqrt(1.0 / float(C // self.num_heads))

        # Compute attention weights with optional attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(q_scaled.dtype)
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        # Compute attention output
        attn_output = torch.bmm(attn_output_weights, v) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)).transpose(0, 1)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return attn_output

