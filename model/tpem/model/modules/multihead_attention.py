import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    
    """
    Multi-headed attention.（多头自注意力机制）
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim              #输入a的维度d_{model}
        self.num_heads = num_heads              #多头自注意力机制的头数h
        self.attn_dropout = attn_dropout        #Dropout的概率p，是对算出来的自注意力矩阵采取Dropout
        self.head_dim = embed_dim // num_heads  #d_k=d_v=d_{model}/h
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5    #softmax函数中要除以√d_k，这里算出1/√d_k

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        #W_q、W_k、W_v都是(embed_dim, embed_dim)的矩阵，将它们3个组合在一起得到了in_proj_weight

        self.register_parameter('in_proj_bias', None)
        #该函数作用和Parameter是一样的，不过它可以用字符串命名参数，即令参数'in_proj_bias'为None
        #而Parameter函数是直接把某个张量传入，这个张量成为一个可以训练的参数

        if bias:                                                    #Q=W_q*I+B
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  #计算b^i的W_o

        if add_bias_kv:                     #对b在第三维上拼上一个维，作为一个偏置项
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        """
        对参数进行初始化。设计者在这里没有直接使用nn.Linear，而是自己设计相关参数并进行初始化
        """
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)        #偏置用常数0进行初始化
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        这里的query, key, value有误导性，实际为计算出query, key, value的x
        1. Input shape: Time * Batch * Channel（序列长度 * batch大小 * 每一个样本的特征数量）
        2. Self-attention can be implemented by passing in the same arguments for query, key and value.
        （即自己对自己进行self-attention的情况）
        3. Timesteps can be masked by supplying a T * T mask in the `attn_mask` argument.
        （masked self-attention使用方法）
        4. Padding elements can be excluded from the key by passing a binary ByteTensor
        (`key_padding_mask`) with shape: batch * src_len, where padding elements are indicated by 1s.
        """
        #data_ptr()返回tensor首元素的内存地址，以此判断qkv、kv是否分别一样
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()      #tgt_len==seq_len，bsz==batch，和transformer那边统一一下
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention（传入的都是x，即Q、K、V应该是要通过x乘上一个权重算出来的）
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention（Q是从x来的，K、V是从另一个x'得来的，通过这个分支可以一次性处理）
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            # Q是从x来的，K是从x'来的，V是从x''来的，只好3者分开处理了
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling                #Q / √d_k

        if self.bias_k is not None:
            assert self.bias_v is not None

            # torch.repeat的用法参见https://blog.csdn.net/qq_29695701/article/details/89763168
            # 因此下述repeat后得到的张量维度为(1, bsz, embed_dim)，正好在K、V的第三维上拼上
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            # 如果k、v有偏置，且需要加mask，则在attn_mask后加上一列的0
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        #使用transpose、view等函数时，原张量在内存中不会被改变，只是输出的时候按照结果的顺序从内存中读
        #使用contiguous()函数可以将张量复制一份，并按照顺序在内存中存放，参考https://blog.csdn.net/weixin_43977640/article/details/111152239
        
        #query==>(seq_len, batch, embed_dim), W_q==>(embed_dim, embed_dim)，q==>(seq_len, batch, embed_dim)
        #这里就可以理解成q先转换为(seq_len, batch*h, head_dim)，再转为(batch*h, seq_len, head_dim)
        #注意有一个关系：embed_dim//h=head_dim，因此可行。k、v同理
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)     #记k的第一个维度（即上面自动算的-1）为src_len（按照一般的transformer的方法，等于seq_len）

        #如果要添加值为0的注意力，设定在行（张量的第一个维度）上添加，k.size()[2:]表示的是new_zeros的第二个维度
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        #torch.bmm实现两个三维（必须是三维）张量的乘法，实际是第一维保留，每个矩阵切片做矩阵乘法
        #q==>(batch*h, seq_len, head_dim), k==>(batch*h, src_len, head_dim),
        #k.transpose(1, 2)==>(batch*h, head_dim, src_len), attn_weights==>(batch*h, seq_len, src_len)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                #注意力权重张量，上三角的部分全部被遮盖为-Inf（虽然attn_mask第0个维度为1，但可以广播到每一层）
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
        
        #softmax的dim=-1表示attn_weights的第二个维度（行）和为1，与论文相符
        #type_as函数使得通过softmax函数的张量也为float张量（和attn_weights数据类型相符）
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)（这两行是原作者注释掉的代码，不是对下面的解释）
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        #attn_weights==>(batch*h, seq_len, src_len), v==>(batch*h, src_len, head_dim)
        #attn==>(batch*h, seq_len, head_dim)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        #attn==>(seq_len, batch, head_dim*h)==(seq_len, batch, embed_dim)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads（attn_weights的第一个维度分出来）
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)        #张量按行分为3块，即Q、K、V

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        """
        这3个函数分别返回对应权值与Q、K、V相乘再加上偏置的结果
        """
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)  #如果在可选参数中未给出weight参数，则使用self.in_proj_weight
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]           #分别取出Q、K、V的那一部分weight，end若为None，效果跟不写一样
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
