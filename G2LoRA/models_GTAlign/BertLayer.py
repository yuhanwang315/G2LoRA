import torch

from typing import Optional, Tuple
from torch import nn
from transformers import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertLayer,BertOutput
from .attention_lora_T import Attention_LoRA
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
# from transformers.modeling_layers import GradientCheckpointingLayer

# class BertLayer(BertLayer):
#     def __init__(self, config,task,n_tasks):
#         super().__init__(config)
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 从配置中获取前向传播时的块大小，用于控制前馈神经网络的分块处理（主要用于节省内存）
#         self.seq_len_dim = 1  # 设置序列长度维度为 1（通常是 batch_size 维度后面的一维）
#         self.attention = Attention_LoRA(            
#             dim=384, 
#             num_heads=8, 
#             r=32, 
#             n_tasks=n_tasks )  # 实例化 BertAttention，用于处理输入的隐藏状态
#         # self.attention = BertAttention(config)
#         self.is_decoder = config.is_decoder  # 用于判断当前层是否是解码器的一部分
#         self.add_cross_attention = config.add_cross_attention  # 用于判断是否需要添加交叉注意力层（通常在解码器中用于处理编码器的输出

#         # 实例化 BertIntermediate 和 BertOutput，分别用于前馈神经网络的中间层和输出层
#         self.intermediate = BertIntermediate(config)  # 初始化中间层
#         self.output = BertOutput(config)  # 初始化输出层
#         self.task=-1

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         output_attentions: Optional[bool] = False,
#         cache_position=None,     
#         **kwargs,    
#     ) -> Tuple[torch.Tensor]:

#         # --------------1. 自注意力机制--------------------------------------------------------------- #
#         # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
#         # 解码器单向自注意力缓存的键/值对位于位置 1 和 2
#         self_attn_past_key_value = past_key_values[:2] if past_key_values is not None else None
#         outputs = self.attention(
#             x=hidden_states,
#             task =  self.task,
#             get_cur_feat=True
#         )
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk,  # 前馈网络的计算函数
#             self.chunk_size_feed_forward,  # 前馈网络计算的块大小
#             self.seq_len_dim,  # 序列长度维度
#             outputs    # 自注意力层的输出
#         )
#         outputs = layer_output + outputs

#         return (outputs,) 


#     # 前馈神经网络
#     def feed_forward_chunk(self, attention_output):
#         # 通过前馈网络的中间层处理自注意力输出
#         intermediate_output = self.intermediate(attention_output)
#         # 将中间层输出和注意力输出传递给前馈网络的输出层
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output

class BertLayer(BertLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.cur_matrix = torch.zeros(384 ,384)
        self.n_cur_matrix = 0

    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(hidden_states.detach().permute(0, 2, 1), hidden_states.detach()).sum(dim=0).cpu()) / (self.n_cur_matrix + hidden_states.shape[0] * hidden_states.shape[1])
        self.n_cur_matrix += hidden_states.shape[0] * hidden_states.shape[1]
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output