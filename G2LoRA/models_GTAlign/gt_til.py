import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from typing import Any, Dict, Optional
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, global_mean_pool, SAGEConv, GATConv, GINConv, SAGPooling
from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from .adapter import Adapter
from .attention_lora import Attention_LoRA

non_MP = SimpleConv(aggr='mean', combine_root='sum')

class LoRAGPSConv(GPSConv):
    def __init__(self, channels: int, conv: torch.nn.Module, heads: int = 8, dropout: float = 0.0,
                 act: str = 'relu', act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[str] = 'batch_norm',
                 norm_kwargs: Optional[Dict[str, Any]] = None, attn_type: str = 'multihead',
                 attn_kwargs: Optional[Dict[str, Any]] = None, r: int = 8,n_tasks: int = 7):
        super().__init__(channels, conv, heads, dropout, act, act_kwargs, norm, norm_kwargs, attn_type="multihead", attn_kwargs=attn_kwargs)

        # Initialize LoRA attention layers instead of multihead attention
        self.attn = Attention_LoRA(
            dim=channels, 
            num_heads=heads, 
            r=r, 
            n_tasks=n_tasks  )
        self.x = None 
        for name, param in self.named_parameters():
            param.requires_grad = False  # Freeze all parameters

        # Unfreeze LoRA-related parameters
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
        
    def forward(self, x, edge_index , batch=None,task = 8 , **kwargs):
        r"""Runs the forward pass of the module."""
        # print("aaa:",x.shape)
        
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
        g_x = global_mean_pool(x, batch)
        if task ==1:
            if self.x is not None  :  # 如果 self.x 已经存在
                self.x = torch.cat((self.x, x), dim=0)  # 拼接当前的 x 到 self.x
            else:
                self.x = x  # 如果不存在，初始化 self.x 为当前的 x
        else:
            if self.x is not None  :  # 如果 self.x 已经存在
                self.x = torch.cat((self.x, g_x), dim=0)  # 拼接当前的 x 到 self.x
            else:
                self.x = g_x  # 如果不存在，初始化 self.x 为当前的 x
        # Global attention transformer-style model.
        # print(len(hs),x.shape)
        h, mask = to_dense_batch(x, batch)
        # print(h.shape, mask.shape,batch)
        # print(mask)
        # print("haha:",h.shape)
        # print(h)
        # print("a:",h.shape)
        h = self.attn(h,self.x , task = task, get_cur_feat=True,key_padding_mask=~mask)
        # print("baba:",h.shape,mask)
        # print("b:",h,h.shape)
        h = h[mask]
        # print("c:",h,h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

class GPS_til(torch.nn.Module):
    def __init__(self, in_dim:int, channels: int, out_dim: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any], r=64,islora=False,n_tasks=7):
        super().__init__()

        self.node_emb = torch.nn.Linear(in_dim, channels - pe_dim)
        self.pe_lin = Linear(32, pe_dim)
        self.pe_norm = BatchNorm1d(32)

        self.convs = ModuleList()
        if islora:
            for l in range(num_layers):
                conv = LoRAGPSConv(channels, SAGEConv(channels,channels), heads=8,
                            attn_type=attn_type, attn_kwargs=attn_kwargs,r=r,n_tasks=n_tasks)
                self.convs.append(conv)
        else:
            for l in range(num_layers):
                conv = GPSConv(channels, SAGEConv(channels,channels), heads=8,
                            attn_type=attn_type, attn_kwargs=attn_kwargs)
                self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels*2, 384),
        )
        self.mlp2 = Sequential(
            Linear(channels, 768),
        )
        self.mlp3 = Sequential(
            Linear(channels, 384),
        )
        # self.attn_pool = SAGPooling(channels, 0.1)
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        self.rank =r
       
        # for name, param in self.named_parameters():
        #     if 'mlp'  in name.lower():
        #         param.requires_grad = False
        # for name, param in self.named_parameters():
        #      param.requires_grad = False
        # 2. 只解冻 LoRA 相关参数
        for name, param in self.named_parameters():
            if 'mlp' in name.lower():
                param.requires_grad = False
    def forward(self, x, pe, edge_index, batch, center_idx ,task):
        # print(x.shape)
        x_pe = self.pe_norm(pe)
        if x.dim() > 2:
            x = x.squeeze(-1)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch,task)
        # print("x",x.shape)
        # mean pool
        g_x = global_mean_pool(x, batch)
        # print("g_x：",g_x.shape)
        c_x = x[center_idx]
        g_x=torch.cat((g_x, c_x), 1) # cat average and center
        w1=self.mlp(g_x)
        w2=self.mlp2(c_x)
        # print("g_x：",g_x.shape)
        return w1,w2

    def graph(self, batch: Batch, task):
        # 用于存储每个图的嵌入
        graph_embeddings = []
        
        for data in batch:
            # 获取每个图的节点特征和位置嵌入

            x = data.x  # 节点特征
            pe = data.pe  # 位置嵌入
            
            # 对位置嵌入进行归一化
            x_pe = self.pe_norm(pe)
            
            # 如果x的维度大于2，压缩多余的维度
            if x.dim() > 2:
                x = x.squeeze(-1)
            
            # 结合节点特征和位置嵌入
            # print(x.shape,x_pe.shape)
            # print(self.node_emb(x).shape,self.pe_lin(x_pe).shape)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            # 使用图卷积层处理每个图
            for conv in self.convs:
                x = conv(x, data.edge_index, data.batch,task)
            
            # 对每个图进行平均池化
            g_x = global_mean_pool(self.mlp3(x), data.batch)

            
            # 传递到MLP得到每个图的最终嵌入
            # graph_embeddings.append(self.mlp3(g_x))
            graph_embeddings.append(g_x)
        
        # 将所有图的嵌入合并为一个批次
        return torch.stack(graph_embeddings, dim=0) 


    def link(self, batch: Batch,task):
        x, edge_index, pe = batch.x, batch.edge_index, batch.pe

        pe_dim = pe.size(1)
        if pe_dim != 32:
            if pe_dim < 32:
                if 32 % pe_dim == 0:
                    repeat_factor = 32 // pe_dim
                    pe = pe.repeat_interleave(repeat_factor, dim=1)
                else:
                    pe_expanded = torch.zeros((pe.size(0), 32), device=pe.device)
                    for i in range(32):
                        orig_idx = int(i * pe_dim / 32)
                        pe_expanded[:, i] = pe[:, orig_idx]
                    pe = pe_expanded
             
            elif pe_dim > 32:
                pe = pe[:, :32]

        x_pe = self.pe_norm(pe)

        if x.dim() > 2:
            x = x.squeeze(-1)
        expected_in_dim = self.node_emb.in_features
    
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            try:
                if hasattr(batch, 'batch'):
                    node_batch_vector = batch.batch
                    # x = conv(x, edge_index, node_batch_vector)
                    x = conv(x, edge_index, node_batch_vector,task)
                else:
                    x = conv(x, edge_index)
            except Exception as e:
                raise e

        x = torch.cat((x, x), 1)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
