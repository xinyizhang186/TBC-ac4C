import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
The TBC-ac4C model includes embedding layer, Transformer encoder, temporal convolutional network (TCN), bidirectional GRU, and 
cross-attention mechanism.
"""

'''Embedding layer: token_Embedding + position_Embedding'''
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_embed = nn.Embedding.from_pretrained(self.position_encoding(max_len, d_model), freeze=True).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, dtype=torch.long, device=device)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        return self.norm(self.tok_embed(x) + self.pos_embed(pos))

    @staticmethod
    def position_encoding(max_len, d_model):
        """
        Position encoding feature introduced in "Attention is all you need",
        the b is changed to 1000 for the short length of sequence.
        """
        b = 1000
        pos_encoding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (b ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (b ** (2 * i / d_model)))
        return torch.FloatTensor(pos_encoding)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() # (batch_size,channels,sequence_length)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01) 
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # L_out = [L_in -d(k-1)-1+2p]/s+1 =[L_in -d(k-1)-1+2p] +1 = L_in + p
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=64, v_dim=64, num_heads=6, p=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        self.drop = nn.Dropout(p)

    def forward(self, x1, x2, mask=None, return_attention=False):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.drop(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        if return_attention:
            return output, attn

        return output


class TBC_ac4C(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 64
        self.max_len = 201 

        self.embedding = Embedding(5, self.emb_dim, self.max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3,norm=nn.LayerNorm(self.emb_dim))

        self.tcv = TemporalConvNet(num_inputs=self.emb_dim, num_channels=[64, 64, 100], kernel_size=5, dropout=0.1)
        self.BiGRU = nn.GRU(input_size=self.emb_dim, hidden_size=50, dropout=0.3, num_layers=2, batch_first=True, bidirectional=True)
        self.cross = CrossAttention(in_dim1=100, in_dim2=100)

        self.shapeChange = nn.Sequential(nn.Linear(self.max_len, 512), nn.LayerNorm(512), nn.GELU())
        self.classifier = nn.Sequential(
             nn.Linear(100, 64), nn.BatchNorm1d(64), nn.Dropout(0.3), nn.LeakyReLU(),
             nn.Linear(64, 2), nn.Softmax(dim=1))

    def forward(self, seqs):
        seqs_feature = self.embedding(seqs)
        seqs_feature = self.transformer_encoder(seqs_feature)

        output1, _ = self.BiGRU(seqs_feature)
        seqs_feature = seqs_feature.permute(0, 2, 1)
        output2 = self.tcv(seqs_feature)
        output2 = output2.permute(0, 2, 1)

        output = self.cross(output1, output2)

        output = torch.transpose(output, 1, 2)
        output = self.shapeChange(output)
        output = nn.functional.max_pool1d(output, output.size(2)).squeeze(2)
        output = self.classifier(output)

        return output



