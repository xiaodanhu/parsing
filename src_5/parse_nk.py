import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

# import prob_tree
# pt=prob_tree.prob_tree()

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_helper3 as chart_helper
import nkutil

# import trees
import treesvideo2 as treesvideo

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

TAG_UNK = "UNK"

PRINT=False

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }

# %%

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        # Note that the torch copy will be on GPU if use_cuda is set
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

# %%

class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None

class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

# %%

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        # NOTE(nikita): the t2t code does the following instead, with eps=1e-6
        # However, I currently have no reason to believe that this difference in
        # implementation matters.
        # mu = torch.mean(z, keepdim=True, dim=-1)
        # variance = torch.mean((z - mu.expand_as(z))**2, keepdim=True, dim=-1)
        # ln_out = (z - mu.expand_as(z)) * torch.rsqrt(variance + self.eps).expand_as(z)
        # ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

# %%

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

# %%

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1)) # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
                ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.bool)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
            )

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded

# %%

class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()


    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

# %%

class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

# %%

class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            num_embeddings_list,
            d_embedding,
            d_positional=None,
            max_len=500,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            emb_dropouts_list=None,
            extra_content_dropout=None,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        embs = []
        emb_dropouts = []
        for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            emb = nn.Embedding(num_embeddings, self.d_content, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)
        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, xs, batch_idxs, extra_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
            ]
        content_annotations = sum(content_annotations)
        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                content_annotations += self.extra_content_dropout(extra_content_annotations, batch_idxs)
            else:
                content_annotations += extra_content_annotations

        timing_signal = torch.cat([self.position_table[:seq_len,:] for seq_len in batch_idxs.seq_lens_np], dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        # Combine the content and timing signals
        if self.partitioned:
            if len(content_annotations) != len(timing_signal):
                aa = 1
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        # TODO(nikita): reconsider the use of layernorm here
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

        return annotations, timing_signal, batch_idxs

# %%

class CharacterLSTM(nn.Module):
    def __init__(self, num_embeddings, d_embedding, d_out,
            char_dropout=0.0,
            normalize=False,
            **kwargs):
        super().__init__()

        self.d_embedding = 2048
        self.d_out = d_out
        self.num_layers = 1

        self.lstm = nn.LSTM(self.d_embedding, self.d_out // 2, num_layers=self.num_layers, bidirectional=True)

        self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
        #TODO(nikita): feature-level dropout?
        self.char_dropout = nn.Dropout(char_dropout)

        if normalize:
            print("This experiment: layer-normalizing after character LSTM")
            self.layer_norm = LayerNormalization(self.d_out, affine=False)
        else:
            self.layer_norm = lambda x: x

    def forward(self, chars_padded_np):
        hidden_state = torch.zeros(self.num_layers * 2, chars_padded_np.size(0), self.d_out // 2).cuda()
        cell_state = torch.zeros(self.num_layers * 2, chars_padded_np.size(0), self.d_out // 2).cuda()

        output, (lstm_out, last_cell_state) = self.lstm(chars_padded_np.unsqueeze(0), (hidden_state, cell_state))

        # print('LSTM ---------------------->', lstm_out.shape)

        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        res = self.layer_norm(lstm_out)
        return res

# %%
def get_elmo_class():
    # Avoid a hard dependency by only importing Elmo if it's being used
    from allennlp.modules.elmo import Elmo
    return Elmo

# %%
def get_bert(bert_model, bert_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    if bert_model.endswith('.tar.gz'):
        tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    bert = BertModel.from_pretrained(bert_model)
    return tokenizer, bert

# %%

class Encoder(nn.Module):
    def __init__(self, embedding,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024,
                    d_positional=None,
                    num_layers_position_only=0,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super().__init__()
        # Don't assume ownership of the embedding as a submodule.
        # TODO(nikita): what's the right thing to do here?
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding

        d_k = d_v = d_kv

        self.stacks = []
        for i in range(num_layers):
            attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

    def forward(self, xs, batch_idxs, extra_content_annotations=None):
        emb = self.embedding_container[0]
        res, timing_signal, batch_idxs = emb(xs, batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn(res, batch_idxs)
            else:
                res, current_attns = attn(res, batch_idxs, qk_inp=timing_signal)
            res = ff(res, batch_idxs)

        return res, batch_idxs


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model, d_ff=1024, num_heads=2, d_kv = 32, 
        dropout_prob=0.1, layer_norm_eps=1e-5,
        d_positional=None, relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1
    ):
        super().__init__()

        d_k = d_v = d_kv

        self.attention1 = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional)
        self.enc_dec_attention = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional)

        self.ff_block = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, tgt, src, batch_idxs):
        x, _ = self.attention1(tgt, batch_idxs)

        if src is not None:
            x, _ = self.enc_dec_attention(x, batch_idxs, src)

        x = self.ff_block(x, batch_idxs)

        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        embedding,
        vocab_size,
        ff_hidden_size,
        d_kv,
        num_blocks=2,
        num_heads=8,
    ):
        super().__init__()

        self.embedding = embedding

        hidden_size = embedding.d_embedding

        self.decoder = []
        for _ in range(num_blocks):
            self.decoder.append(DecoderBlock(hidden_size, ff_hidden_size, num_heads, d_kv))

        self.decoder = nn.ModuleList(self.decoder)

        self.lin_final = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, batch_idxs, src_enc):

        tgt = self.embedding([tgt], batch_idxs)[0]

        for block in self.decoder:
            tgt = block(tgt, src_enc, batch_idxs)

        out = self.lin_final(tgt)

        return out


# %%

class NKChartParser(nn.Module):
    # We never actually call forward() end-to-end as is typical for pytorch
    # modules, but this inheritance brings in good stuff like state dict
    # management.
    def __init__(
            self,
            activity_vocab,
            phrase_vocab,
            action_vocab,
            ph_tree_ref,
            hparams,
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.activity_vocab = activity_vocab
        self.phrase_vocab = phrase_vocab
        self.action_vocab = action_vocab
        self.ph_tree_ref = ph_tree_ref

        self.d_model = hparams.d_model
        self.partitioned = hparams.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (hparams.d_model // 2) if self.partitioned else None

        num_embeddings_map = {
            'tags': activity_vocab.size(),
        }
        emb_dropouts_map = {
            'tags': hparams.tag_emb_dropout,
        }

        self.emb_types = []
        self.emb_types.append('tags')

        self.use_tags = hparams.use_tags

        self.morpho_emb_dropout = None
        if hparams.use_chars_lstm or hparams.use_elmo or hparams.use_bert or hparams.use_bert_only:
            self.morpho_emb_dropout = hparams.morpho_emb_dropout
        else:
            assert self.emb_types, "Need at least one of: use_tags, use_words, use_chars_lstm, use_elmo, use_bert, use_bert_only"

        self.char_encoder = None
        self.elmo = None
        self.bert = None
        if hparams.use_chars_lstm:
            assert not hparams.use_elmo, "use_chars_lstm and use_elmo are mutually exclusive"
            assert not hparams.use_bert, "use_chars_lstm and use_bert are mutually exclusive"
            assert not hparams.use_bert_only, "use_chars_lstm and use_bert_only are mutually exclusive"
            self.char_encoder = CharacterLSTM(
                num_embeddings_map['tags'],
                hparams.d_char_emb,
                self.d_content,
                char_dropout=hparams.char_lstm_input_dropout,
            )
        elif hparams.use_elmo:
            assert not hparams.use_bert, "use_elmo and use_bert are mutually exclusive"
            assert not hparams.use_bert_only, "use_elmo and use_bert_only are mutually exclusive"
            self.elmo = get_elmo_class()(
                options_file="data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                weight_file="data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                num_output_representations=1,
                requires_grad=False,
                do_layer_norm=False,
                keep_sentence_boundaries=True,
                dropout=hparams.elmo_dropout,
                )
            d_elmo_annotations = 1024

            # Don't train gamma parameter for ELMo - the projection can do any
            # necessary scaling
            self.elmo.scalar_mix_0.gamma.requires_grad = False

            # Reshapes the embeddings to match the model dimension, and making
            # the projection trainable appears to improve parsing accuracy
            self.project_elmo = nn.Linear(d_elmo_annotations, self.d_content, bias=False)
        elif hparams.use_bert or hparams.use_bert_only:
            self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case)
            if hparams.bert_transliterate:
                from transliterate import TRANSLITERATIONS
                self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
            else:
                self.bert_transliterate = None

            d_bert_annotations = self.bert.pooler.dense.in_features
            self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

            if hparams.use_bert_only:
                self.project_bert = nn.Linear(d_bert_annotations, hparams.d_model, bias=False)
            else:
                self.project_bert = nn.Linear(d_bert_annotations, self.d_content, bias=False)

        if not hparams.use_bert_only:
            self.embedding = MultiLevelEmbedding(
                [num_embeddings_map[emb_type] for emb_type in self.emb_types],
                hparams.d_model,
                d_positional=self.d_positional,
                dropout=hparams.embedding_dropout,
                timing_dropout=hparams.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=hparams.sentence_max_len,
            )

            self.encoder = Encoder(
                self.embedding,
                num_layers=hparams.num_layers,
                num_heads=hparams.num_heads,
                d_kv=hparams.d_kv,
                d_ff=hparams.d_ff,
                d_positional=self.d_positional,
                num_layers_position_only=hparams.num_layers_position_only,
                relu_dropout=hparams.relu_dropout,
                residual_dropout=hparams.residual_dropout,
                attention_dropout=hparams.attention_dropout,
            )
        else:
            self.embedding = None
            self.encoder = None

        # to predict the sequence
        # self.f_label = nn.Sequential(
        #     nn.Linear(hparams.d_model, hparams.d_label_hidden),
        #     LayerNormalization(hparams.d_label_hidden),
        #     nn.ReLU(),
        #     nn.Linear(hparams.d_label_hidden, frame_vocab.size()),
        #     )

        # predict the action labels
        if hparams.decode_method == 'linear':
            self.tag_action = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, action_vocab.size()),
                )
        elif hparams.decode_method == 'transformer':
            self.embedding_decode = MultiLevelEmbedding(
                [action_vocab.size()],
                hparams.d_model,
                d_positional=512,
                dropout=hparams.embedding_dropout,
                timing_dropout=hparams.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=hparams.sentence_max_len,
            )
            self.tag_action = Decoder(
                embedding=self.embedding_decode,
                vocab_size=action_vocab.size(),
                ff_hidden_size=hparams.d_ff,
                d_kv=hparams.d_kv
            )
            # from model.decoder import Decoder
            # self.tag_action = Decoder(action_vocab.size(), hparams.sentence_max_len, hparams.d_model, hparams.d_ff).cuda()
        elif hparams.decode_method == 'lstm':
            ## TBD
            raise NotImplementedError

        self.tag_action_loss_scale = hparams.tag_loss_scale


        hierarchy = True
        if hierarchy:
            # predict phrase labels
            self.tag_phrase = nn.Sequential(
                nn.Linear(action_vocab.size(), hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, phrase_vocab.size()),
                )
            self.tag_phrase_loss_scale = hparams.tag_loss_scale

            # predict activity labels
            self.tag_activity = nn.Sequential(
                nn.Linear(phrase_vocab.size(), hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, activity_vocab.size()),
                )
            self.tag_activity_loss_scale = hparams.tag_loss_scale

        else:
            # predict phrase labels
            self.tag_phrase = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, phrase_vocab.size()),
                )
            self.tag_phrase_loss_scale = hparams.tag_loss_scale

            # predict activity labels
            self.tag_activity = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, activity_vocab.size()),
                )
            self.tag_activity_loss_scale = hparams.tag_loss_scale

        if use_cuda:
            self.cuda()
        

        # import eval
        # self.evaluator=eval.mAP_Evaluator(self.action_vocab.indices)

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'use_chars_concat' in hparams and hparams['use_chars_concat']:
            raise NotImplementedError("Support for use_chars_concat has been removed")
        if 'sentence_max_len' not in hparams:
            hparams['sentence_max_len'] = 300
        if 'use_elmo' not in hparams:
            hparams['use_elmo'] = False
        if 'elmo_dropout' not in hparams:
            hparams['elmo_dropout'] = 0.5
        if 'use_bert' not in hparams:
            hparams['use_bert'] = False
        if 'use_bert_only' not in hparams:
            hparams['use_bert_only'] = False
        if 'predict_tags' not in hparams:
            hparams['predict_tags'] = False
        if 'bert_transliterate' not in hparams:
            hparams['bert_transliterate'] = ""

        spec['hparams'] = nkutil.HParams(**hparams)
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        if not hparams['use_elmo']:
            res.load_state_dict(model)
        else:
            state = {k: v for k,v in res.state_dict().items() if k not in model}
            state.update(model)
            res.load_state_dict(state)
        if use_cuda:
            res.cuda()
        return res

    def split_batch(self, sentences, golds, subbatch_max_tokens=3000):
        if self.bert is not None:
            lens = [
                len(self.bert_tokenizer.tokenize(' '.join([word for (_, word) in sentence]))) + 2
                for sentence in sentences
            ]
        else:
            lens = [len(sentence) + 2 for sentence in sentences]

        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse(self, sentence, gold=None):
        tree_list, loss_list = self.parse_batch([sentence], [gold] if gold is not None else None)
        return tree_list[0], loss_list[0]

    def parse_batch(self, sentences, golds=None, return_label_scores_charts=False):
        is_train = golds is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)

        if golds is None:
            golds = [None] * len(sentences)

        # print("Sentence length ---------------> ", len(sentences[0]))

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            for (actn_tag, ph_tag, actv_tag, word) in [(START, START, START, START)] + sentence + [(STOP, STOP, STOP, STOP)]:
                batch_idxs[i] = snum
                # tag_idxs[i] = self.action_vocab.index(actn_tag)
                i += 1

        batch_idxs_less = BatchIndices(batch_idxs[:-1])
        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs
        }
        emb_idxs = [
            from_numpy(emb_idxs_map[emb_type])
            for emb_type in self.emb_types
            ]

        if is_train and self.tag_action is not None:
            gold_tag_idxs = from_numpy(emb_idxs_map['tags'])

        # print("Gold ind ", gold_tag_idxs)

        extra_content_annotations = None
        max_word_len = max([max([len(word) for _, _, _, word in sentence]) for sentence in sentences])
        # Add 2 for start/stop tokens
        max_word_len = max(max_word_len, 3) + 2
        # frame_embeds = torch.vstack([torch.stack([word for tag, word in sentence]) for sentence in sentences]).cuda()

        frame_embeds = []
        actn_tags = []
        ph_tags = []
        actv_tags = []
        gold_tag_action_idxs = []
        gold_tag_phrase_idxs = []
        gold_tag_activity_idxs = []
        for snum, sentence in enumerate(sentences):
            for wordnum, (actn_tag, ph_tag, actv_tag, word) in enumerate([(START, START, START, START)] + sentence + [(STOP, STOP, STOP, STOP)]):

                actn_tags.append(actn_tag)
                ph_tags.append(ph_tag)
                actv_tags.append(actv_tag)
                gold_tag_action_idxs.append(self.action_vocab.index(actn_tag))
                gold_tag_phrase_idxs.append(self.phrase_vocab.index(ph_tag))
                gold_tag_activity_idxs.append(self.activity_vocab.index(actv_tag))

                if word in (START, STOP):
                    frame_embeds.append(torch.zeros((2048)))
                    # gold_tag_activity_idxs.append(self.activity_vocab.index(actv_tag))
                else:
                    frame_embeds.append(word)
                    # gold_tag_activity_idxs.append(self.activity_vocab.index(actv_tag[0]))


        if PRINT:
            print("Gold ind action", gold_tag_action_idxs)
            print("Gold ind phrase", gold_tag_phrase_idxs)
            print("Gold ind activity", gold_tag_activity_idxs)
        gold_tag_action_idxs = torch.tensor(gold_tag_action_idxs).cuda()
        gold_tag_phrase_idxs = torch.tensor(gold_tag_phrase_idxs).cuda()
        gold_tag_activity_idxs = torch.tensor(gold_tag_activity_idxs).cuda()
        frame_embeds = torch.vstack(frame_embeds).cuda()

        if PRINT:
            print("Dictionary Action size ", self.action_vocab.size())
            print("Dictionary Phrase size ", self.frame_vocab.size())

            print(frame_embeds.shape)
            print("Gold ind ", gold_tag_idxs)

        extra_content_annotations = self.char_encoder(frame_embeds)

        #
        # print("Tags:", tags[2])
        #
        # print("ECA", extra_content_annotations.shape)

        annotations, _ = self.encoder(emb_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)


        if PRINT: print("Annotations ", annotations.shape)


        if self.partitioned:
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            annotations = torch.cat([
                annotations[:, 0::2],
                annotations[:, 1::2],
            ], 1)

        # if self.f_tag is not None:
        tag_annotations = annotations

        fencepost_annotations = torch.cat([
            annotations[:-1, :self.d_model//2],
            annotations[1:, self.d_model//2:],
            ], 1)
        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations


        # if self.f_tag is not None:
        if self.spec['hparams']['decode_method'] == 'linear':
            tag_logits_action = self.tag_action(tag_annotations)
        elif self.spec['hparams']['decode_method'] == 'transformer':
            if not self.spec['hparams']['is_inference']:
                tgt = gold_tag_action_idxs[:-1]  # save start token, skip end token
                gold_tag_action_idxs, gold_tag_phrase_idxs, gold_tag_activity_idxs = gold_tag_action_idxs[1:], gold_tag_phrase_idxs[1:], gold_tag_activity_idxs[1:]
                tag_logits_action = self.tag_action(tgt, batch_idxs_less, tag_annotations[:-1])
                # tag_logits_action = self.tag_action(gold_tag_action_idxs, batch_idxs, tag_annotations)
            else:
                output_tokens = (torch.ones((tag_annotations.shape[0])) * self.action_vocab.index(STOP)).type_as(tag_annotations).long() # (B, max_length)
                output_tokens[0] = self.action_vocab.index(START)  # Set start token
                for Sy in range(1, tag_annotations.shape[0]):
                    # y = output_tokens[:Sy]  # (B, Sy)
                    output = self.tag_action(output_tokens, batch_idxs, tag_annotations)  # (Sy, B, C)
                    output = torch.argmax(output, dim=-1)  # (Sy, B)
                    output_tokens[Sy] = output[-1]  # Set the last output token

                self.evaluator.evaluate([gold_tag_action_idxs.cpu().numpy()],[output_tokens.cpu().numpy()])
                return None, None, self.evaluator.mAP, None

        tag_loss_action =  nn.functional.cross_entropy(tag_logits_action, gold_tag_action_idxs, reduction='sum')
        if PRINT: print("------> Action Tag Loss: ", tag_loss_action)

        hierarchy = False
        if hierarchy:
            tag_logits_phrase = self.tag_phrase(tag_logits_action)
            tag_loss_phrase = nn.functional.cross_entropy(tag_logits_phrase, gold_tag_phrase_idxs, reduction='sum')
            if PRINT: print("------> Phrase Tag Loss: ", tag_loss_phrase)

            tag_logits_activity = self.tag_activity(tag_logits_phrase)
            tag_loss_activity = nn.functional.cross_entropy(tag_logits_activity, gold_tag_activity_idxs, reduction='sum')
            if PRINT: print("------> Activity Tag Loss: ", tag_loss_activity)

        else:
            tag_logits_phrase = self.tag_phrase(tag_annotations)
            tag_loss_phrase = nn.functional.cross_entropy(tag_logits_phrase, gold_tag_phrase_idxs, reduction='sum')
            if PRINT: print("------> Phrase Tag Loss: ", tag_loss_phrase)

            tag_logits_activity = self.tag_activity(tag_annotations)
            tag_loss_activity = nn.functional.cross_entropy(tag_logits_activity, gold_tag_activity_idxs, reduction='sum')
            if PRINT: print("------> Activity Tag Loss: ", tag_loss_activity)

        predictions_action = torch.max(tag_logits_action, 1)[1].to('cuda')
        predictions_phrase = torch.max(tag_logits_phrase, 1)[1].to('cuda')
        predictions_activity = torch.max(tag_logits_activity, 1)[1].to('cuda')
        correct_action = float(torch.sum(predictions_action == gold_tag_action_idxs)) / float(len(predictions_action))
        correct_phrase = float(torch.sum(predictions_phrase == gold_tag_phrase_idxs)) / float(len(predictions_phrase))
        correct_activity = float(torch.sum(predictions_activity == gold_tag_activity_idxs)) / float(len(predictions_activity))

        self.evaluator.evaluate([gold_tag_action_idxs.cpu().numpy()],[predictions_action.cpu().numpy()])
        # if self.evaluator.mAP > 0.1:
        #     print(self.evaluator.mAP)

        # Note that the subtraction above creates fenceposts at sentence
        # boundaries, which are not used by our parser. Hence subtract 1
        # when creating fp_endpoints
        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if PRINT:
            print("Start point actions", fp_startpoints)
            print("Start point actions", fp_endpoints)

        # Just return the charts, for ensembling
        # print("return_label_scores_charts----> ", return_label_scores_charts)
        # chart = self.(annotations)
        # print("------> Chart: ", chart.shape)

        def unique_order(list):
            output = []
            for i in list:
                if (i!=self.action_vocab.index(START)) & (i!=self.action_vocab.index(STOP)):
                    if (int(i) not in output):
                        output.append(int(i))
            # print(output)
            return output

        def unique_order_repeated(list):
            output = []
            previous=-1
            for i in list:
                if (i!=self.action_vocab.index(START)) & (i!=self.action_vocab.index(STOP)):
                    # if (int(i) not in output):
                    if previous == -1:
                        previous = i
                        output.append(int(i))
                    if i !=previous:
                        previous = i
                        output.append(int(i))
            # print(output)
            return output

        def unique_order_ph(list):
            output = []
            for i in list:
                if (i!=START) & (i!=STOP):
                    if (i not in output):
                        output.append(i)
            # print(output)
            return output

        def phrase_intervals(list):
            start = [0]
            end = []
            previous = '<START>'
            for k, i in enumerate(list):
                if i!=START and i!=STOP:
                    if previous == '<START>':
                        previous = i
                    if i !=previous:
                            previous = i
                            end.append(k-1)
                            if k !=len(list):
                                start.append(k)

            if not len(list)-1 in end:
                end.append(len(list)-1)
            # print(list)
            # print(start[:-1])
            # print(end)
            return start[:],end

        return_label_scores_charts=False
        fp_startpoints, fp_endpoints = phrase_intervals(ph_tags)
        if PRINT:
            print("Start point phrase", fp_startpoints)
            print("Start point phrase", fp_endpoints)

        # exit()
        loss = 0.0
        if return_label_scores_charts:
            charts = []
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                chart = self.label_scores_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:])
                phrase_prob = torch.sigmoid(torch.mean(chart.squeeze(), 0))
                # print("------> Chart: ", chart.shape)
                # print("------> Word", torch.argmax(chart.squeeze(), -1).cpu())
                # print("------> Mean Phrase", torch.sigmoid(torch.mean(chart.squeeze(), 0)).cpu())
                # print("------> Dictionary Phrase", self.frame_vocab.size())
                per_sentence_tag_idxs = torch.argmax(chart, -1).cpu() # per frame
                # per_sentence_tag_idxs = torch.argmax(chart).cpu()
                if PRINT:
                    print("=========> Gold Tags", (tags[start:end]))
                    print("=========> Gold Phrase", (ph_tags[start:end]))
                # print("===========>", self.frame_vocab.value(per_sentence_tag_idxs))
                per_sentence_tags = [[self.frame_vocab.value(idx) for idx in idxs[1:-1]] for idxs in per_sentence_tag_idxs]
                if PRINT: print("------> Predicted Phrases: ", per_sentence_tags)
                charts.append(chart.cpu().data.numpy())

                # print("-------> Unique Tags:", np.unique(tags[start+1:end]))
                # print("-------> Unique Tags:", torch.unique(gold_tag_idxs[start+1:end]).cpu().numpy())
                # print("-------> Unique Ph Tags:", np.unique(ph_tags[start+1:end]))
                # print("-------> Unique Ph Tags:", list(set(ph_tags)))
                # print("-------> All Ph Tags:", ph_tags)
                # print("-------> Unique Ph Tags:", list(set(tags)))
                # print()

                # action_seq = unique_order(gold_tag_idxs[start+1:end])
                action_seq = unique_order_repeated(gold_tag_idxs[start+1:end])
                tags_tree_prob=[]
                for key in self.ph_tree_ref:
                    # print(key)
                    # print(self.ph_tree_ref[key])
                    try:
                        tags_tree_prob.append(pt.p2action(key, action_seq))
                    except:
                        tags_tree_prob.append(0)
                tags_tree_prob = torch.Tensor(tags_tree_prob).cuda()
                pr_loss = nn.functional.l1_loss(phrase_prob[:-2], tags_tree_prob, reduction='mean')
                loss+=pr_loss

                if PRINT:
                    print("-------> Unique Tags IDs:", action_seq )
                    print("-------> Unique Tags:", unique_order_ph(tags[start+1:end]))
                    print("-------> Unique Ph Tags:", unique_order_ph(ph_tags[start+1:end]))
                    print("-------> Probability of Ph Tags:", tags_tree_prob)
                    print("-------> Probability Loss:", pr_loss)
                    print()
                    print('----------------------------------------------------')

        # total_loss = tag_loss_action + torch.exp(torch.tensor([1])).to('cuda') * tag_loss_phrase + torch.exp(torch.tensor([2])).to('cuda') * tag_loss_activity
        # total_loss = 1 * tag_loss_action + 6.7 * tag_loss_phrase + 2.4 * tag_loss_activity
        total_loss = 1 * tag_loss_action + 6.7 * tag_loss_phrase + 2.4 * tag_loss_activity
        
        return (correct_action, correct_phrase, correct_activity), total_loss, self.evaluator.mAP, predictions_action.cpu().numpy(), gold_tag_action_idxs.cpu().numpy()
       

        if not is_train:
            # trees = []
            # scores = []
            # if self.f_tag is not None:
                # Note that tag_logits includes tag predictions for start/stop tokens
                tag_idxs = torch.argmax(tag_logits, -1).cpu()
                per_sentence_tag_idxs = torch.split_with_sizes(tag_idxs, [len(sentence) + 2 for sentence in sentences])
                per_sentence_tags = [[self.action_vocab.value(idx) for idx in idxs[1:-1]] for idxs in per_sentence_tag_idxs]

        #     for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
        #         sentence = sentences[i]
        #         if self.f_tag is not None:
        #             sentence = list(zip(per_sentence_tags[i], [x[1] for x in sentence]))
        #         tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentence, golds[i])
        #         trees.append(tree)
        #         scores.append(score)
        #     return trees, scores
        #
        # # During training time, the forward pass needs to be computed for every
        # # cell of the chart, but the backward pass only needs to be computed for
        # # cells in either the predicted or the gold parse tree. It's slightly
        # # faster to duplicate the forward pass for a subset of the chart than it
        # # is to perform a backward pass that doesn't take advantage of sparsity.
        # # Since this code is not undergoing algorithmic changes, it makes sense
        # # to include the optimization even though it may only be a 10% speedup.
        # # Note that no dropout occurs in the label portion of the network
        # pis = []
        # pjs = []
        # plabels = []
        # paugment_total = 0.0
        # num_p = 0
        # gis = []
        # gjs = []
        # glabels = []
        # with torch.no_grad():
        #     for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
        #         p_i, p_j, p_label, p_augment, g_i, g_j, g_label = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentences[i], golds[i])
        #         paugment_total += p_augment
        #         num_p += p_i.shape[0]
        #         pis.append(p_i + start)
        #         pjs.append(p_j + start)
        #         gis.append(g_i + start)
        #         gjs.append(g_j + start)
        #         plabels.append(p_label)
        #         glabels.append(g_label)
        #
        # cells_i = from_numpy(np.concatenate(pis + gis))
        # cells_j = from_numpy(np.concatenate(pjs + gjs))
        # cells_label = from_numpy(np.concatenate(plabels + glabels))
        #
        # cells_label_scores = self.f_label(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
        #
        # '''
        # cells_label_scores = torch.cat([
        #             cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
        #             cells_label_scores
        #             ], 1)
        # '''
        #
        # cells_scores = torch.gather(cells_label_scores, 1, cells_label[:, None].type(torch.int64))
        # loss = cells_scores[:num_p].sum() - cells_scores[num_p:].sum() + paugment_total
        # print("--------------------> Loss", loss)
        # print("-------------------->", paugment_total)
        #
        #
        # # trees = []
        # # scores = []
        # # for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
        # #     sentence = sentences[i]
        # #     tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentence)
        # #     trees.append(tree)
        # #     scores.append(score)
        #
        # if self.f_tag is not None:
        #     return None, (loss, tag_loss)
        # else:
        #     return None, loss

    def label_scores_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end):
        # Note that the bias added to the final layer norm is useless because
        # this subtraction gets rid of it

        # print("End ", fencepost_annotations_end.shape)
        # print("End ", torch.unsqueeze(fencepost_annotations_start, 1).shape)
        # span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
        #                  - torch.unsqueeze(fencepost_annotations_start, 1))

        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 0))

        label_scores_chart = self.f_label(span_features)
        '''
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
            ], 2)
        '''
        return label_scores_chart

    def parse_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end, sentence, gold=None):
        is_train = gold is not None
        label_scores_chart = self.label_scores_from_annotations(fencepost_annotations_start, fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                n_frame=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.frame_vocab,
                is_train=is_train,
                important_node=self.frame_vocab.important_node)

            # score, p_i, p_j, p_label, _ = chart_helper.decode(False, len(sentence), label_scores_chart_np, False, None, self.frame_vocab)
            p_score, p_i, p_j, p_label, p_augment = chart_helper.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = chart_helper.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            return self.decode_from_chart(sentence, label_scores_chart_np)

    def decode_from_chart_batch(self, sentences, charts_np, golds=None):
        trees = []
        scores = []
        if golds is None:
            golds = [None] * len(sentences)
        for sentence, chart_np, gold in zip(sentences, charts_np, golds):
            tree, score = self.decode_from_chart(sentence, chart_np, gold)
            trees.append(tree)
            scores.append(score)
        return trees, scores

    def decode_from_chart(self, sentence, chart_np, gold=None):
        decoder_args = dict(
            n_frame=len(sentence),
            label_scores_chart=chart_np,
            gold=gold,
            label_vocab=self.frame_vocab,
            is_train=False,
            important_node=self.frame_vocab.important_node)

        force_gold = (gold is not None)

        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.
        score, p_i, p_j, p_label, _ = chart_helper.decode(force_gold, **decoder_args)
        last_splits = []
        idx = -1
        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.frame_vocab.value(label_idx)
            # if not label: # TODO: Remove to get n-ary trees.
            #     label = ('-NONE-',)
            if (i + 1) >= j:
                tag, I3D = sentence[i]
                tree = treesvideo.LeafParseNode_alt(int(i), tag, I3D)
                if type(label)==tuple and label!=():
                    tree = treesvideo.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if type(label)==tuple and label!=():
                    return [treesvideo.InternalParseNode(label, children)]
                else:
                    return children

        tree = make_tree()[0]
        return tree, score
