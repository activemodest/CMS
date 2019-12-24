'''statement_cnn,meta+history_encoder,cat'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformer_1 import Encoder



class Net(nn.Module):
    def __init__(self,
                 word2vec_preweight,
                 vocabulary_dim,
                 index,
                 transformer_num_layers=3,
                 num_heads=3,
                 dropout=0.8,
                 vec_size=300,
                 history_num = 5,
                 final_dim = 300,
                 in_channels=1,
                 out_channels=50,
                 kernel_size=[2, 3, 4],
                 hidden_size = 64,
                 meta_size = 7,
                 class_num = 6):
        super(Net,self).__init__()

        self.word2vec_preweight = word2vec_preweight
        self.vocabulary_dim = vocabulary_dim
        self.vec_size = vec_size
        self.meta_size = meta_size
        self.history_num = history_num
        self.index = index
        self.hidden_size = hidden_size
        self.final_dim = final_dim

        self.embedding = nn.Embedding(self.vocabulary_dim,self.vec_size)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec_preweight))

        # for statement
        self.statement_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (kernel_size[i], vec_size)).cuda() for i in range(len(kernel_size))])  # 1*w*d

        self.Linear_history = nn.Linear(self.history_num, self.final_dim).cuda()

        self.encoder_meta = Encoder(
            max_seq_len=7,
            num_layers=transformer_num_layers,
            model_dim=self.vec_size,
            num_heads=num_heads,
            ffn_dim=self.vec_size * 1 ,
            dropout=dropout).cuda()

        self.dropout = nn.Dropout(dropout).cuda()

        #full connected network
        # self.out = nn.Sequential(
        #     nn.Linear(out_channels*3+self.vec_size,128).cuda(),
        #     nn.Linear(128, 64).cuda(),
        #     nn.ReLU().cuda(),
        #     nn.Linear(64, 32).cuda(),
        #     nn.ReLU().cuda(),
        #     nn.Linear(32, class_num).cuda()
        # )
        self.out = nn.Sequential(nn.Linear(out_channels * 3 + self.vec_size, class_num).cuda())
    def forward(self,statement,statement_len,meta,history,test_label=False):
        #statement_len记录statement的长度
        statement_vec = self.embedding(statement).unsqueeze(1)  # N*W*D
        meta_vec = self.embedding(meta)           # N*W*D
        history_vec = self.Linear_history(history).unsqueeze(1)

        side_information = torch.cat((meta_vec,history_vec),1)
        side_information = side_information[:, self.index, :]

        statement_vector = [F.relu(conv(statement_vec)).squeeze(3) for conv in self.statement_convs]  # [(N,output_channel,W-kernel_size[0]+1),(N,output_channel,W-kernel_size[1]+1)...]
        statement_vector = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in statement_vector]  # [(N,output_channel),(N,output_channel)]
        statement_vector = torch.cat(statement_vector, 1)  # (N,3*output_channel)

        side_input_len = torch.Tensor([7]).int().expand(side_information.size(0))
        side_vec, side_attention = self.encoder_meta(side_information, side_input_len, side_information.size(2),test_label=test_label)
        side_vec = torch.max(side_vec, 1)[0]

        new_final_vector = torch.cat((statement_vector,side_vec),1).cuda()


        new_final_vector = self.dropout(new_final_vector)
        return self.out(new_final_vector)



