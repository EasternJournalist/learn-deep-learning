import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Reference https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb

maxlen = 128 # maximum of length
batch_size = 6
max_pred = 5  # max tokens of prediction
num_layers = 6 # number of Encoder of Encoder Layer
num_heads = 12 # number of heads in Multi-Head Attention
dim_model = 768 # Embedding Size
dim_ff = 3072  # Feed forward 
dim_k = dim_v = 64  # dimension of K(=Q), V
num_segments = 2

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim_model)
        self.pos_embed = nn.Embedding(maxlen, dim_model) 
        self.seg_embed = nn.Embedding(num_segments, dim_model)  
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)              # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dim_k) 
        scores.masked_fill_(attn_mask, -1e9) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(dim_model, dim_k * num_heads)
        self.W_K = nn.Linear(dim_model, dim_k * num_heads)
        self.W_V = nn.Linear(dim_model, dim_v * num_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size , len_q , dim_model], k: [batch_size x len_k x dim_model], v: [batch_size x len_k x dim_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # (batch_size, num_heads, len_q, dim_k)
        q_s = self.W_Q(Q).view(batch_size, -1, num_heads, dim_k).transpose(1,2)  
        k_s = self.W_K(K).view(batch_size, -1, num_heads, dim_k).transpose(1,2) 
        v_s = self.W_V(V).view(batch_size, -1, num_heads, dim_v).transpose(1,2) 

        # batch_size, num_heads, len_q, len_k
        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1) 

        # context: [batch_size x num_heads x len_q x dim_v], attn: [batch_size x num_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * dim_v) # context: [batch_size x len_q x num_heads * dim_v]
        output = nn.Linear(num_heads * dim_v, dim_model)(context)
        return nn.LayerNorm(dim_model)(output + residual), attn # output: [batch_size x len_q x dim_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        # (batch_size, len_seq, dim_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, dim_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x dim_model]
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
        self.fc = nn.Linear(dim_model, dim_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(dim_model, dim_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(dim_model)
        self.classifier = nn.Linear(dim_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, dim_model], attn : [batch_size, num_heads, d_mode, dim_model]
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, dim_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, dim_model]

        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, dim_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf