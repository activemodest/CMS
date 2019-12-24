'''initial transformer'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    '''Scaled dot-product attention mechanism.'''

    def __init__(self, attention_dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout).cuda()
        self.softmax = nn.Softmax(dim=2).cuda()

    def forward(self, q, k, v, scale=None, attn_mask=None,statement_label=False):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if statement_label:
            attention = attention.masked_fill_(attn_mask, -np.inf)  
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads).cuda()
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads).cuda()
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads).cuda()

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.layer_norm = nn.LayerNorm(model_dim).cuda()

    def forward(self, key, value, query, attn_mask=None,statement_label=False):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # slpit by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if statement_label:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)  
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask,statement_label)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        self.max_seq_len = max_seq_len
        position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])
        position_encoding = torch.from_numpy(position_encoding).float()
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self,input_len,statement_label=False):
        if statement_label:
            max_len = torch.max(torch.Tensor([31,6,5])).int()
        else:
            max_len = torch.max(input_len)
        max_len = self.max_seq_len
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        len2 = max_len-input_len[1].cpu()
        input_pos = tensor(
            [list(range(1, len1 + 1)) + [0] * ((max_len - len1.cpu().data.numpy()))  for len1 in input_len])
        return self.position_encoding(input_pos.cuda())

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1).cuda()
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.layer_norm = nn.LayerNorm(model_dim).cuda()

    def forward(self, x):
        output = x.transpose(1, 2)  # [b,w,d]->[b,d,w]
        output = F.relu(self.w1(output)).cuda()
        output = self.w2(output)
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout).cuda()
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout).cuda()

    def forward(self, inputs, attn_mask=None,statement_label=False):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask,statement_label)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def fixedPositionEmbedding(batchSize, sequenceLen=3, model_dim=512):     
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(model_dim)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)

    embeddedPosition = np.array(embeddedPosition, dtype='float32')
    return torch.Tensor(embeddedPosition)


class Encoder(nn.Module):
    def __init__(self,
                 # vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

        #self.seq_embedding = nn.Embedding(max_seq_len,model_dim,padding_idx=0)

        self.pos_embedding = PositionalEncoding(model_dim,max_seq_len)

    def forward(self, inputs_embedding,inputs_len,model_dim=512,inputs=None,statement_label=False,test_label=False):
        #inputs = [B,L]
        pos_embedding1 = self.pos_embedding(inputs_len, statement_label).cuda()
        #print(inputs_embedding.size(),pos_embedding1.size())
        # output = inputs_embedding + pos_embedding1
        output = inputs_embedding

        if statement_label:
            self_attention_mask = padding_mask(inputs,inputs)       #mask padding
            attentions = []
            for encoder in self.encoder_layers:
                output, attention = encoder(output,self_attention_mask,statement_label)
                attentions.append(attention)
        else:
            attentions = []
            for encoder in self.encoder_layers:
                output, attention = encoder(output)
                # output, attention = encoder(output)
                attentions.append(attention)

        if test_label:
            with open('./attention.txt','w') as f_attention:   
                for i in range(len(attentions[-1])):
                    f_attention.write(str(attentions[-1][i])+'\r\n')

        return output, attentions






