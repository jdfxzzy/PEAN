import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        _, _, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        _, _, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3) 
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3) 
        self.fuse_out = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()          
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()     
        x = x.view(B, C, H, W)                                        

        x_input = self.conv_input(x)                                   
        feature_h = x_input.permute(0, 2, 3, 1).contiguous()           
        feature_h = feature_h.view(B * H, W, C)                        
        feature_v = x_input.permute(0, 3, 2, 1).contiguous()            
        feature_v = feature_v.view(B * W, H, C)                        
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)      
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)    
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        attention_output_h = self.attn(q_h, k_h, v_h)              
        attention_output_v = self.attn(q_v, k_v, v_v)              
        attention_output_h = attention_output_h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()   
        attention_output_v = attention_output_v.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()    
        attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))        

        x = attn_out + h                                       
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()    
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)                                       
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)                                

        return x

class Inter_SA(nn.Module):
    def __init__(self,dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size, 3 * self.hidden_size, kernel_size=1, padding=0)  
        self.conv_v = nn.Conv2d(self.hidden_size, 3 * self.hidden_size, kernel_size=1, padding=0)  
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(2*self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()                  
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()              
        x = x.view(B, C, H, W)                                               

        x_input = self.conv_input(x)                                            
        feature_h = torch.chunk(self.conv_h(x_input), 3, dim=1)               
        feature_v = torch.chunk(self.conv_v(x_input), 3, dim=1)                
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)        
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous() 
        horizontal_groups = horizontal_groups.view(3*B, H, -1)                 
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)         
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]


        attention_output_h = self.attn(query_h, key_h, value_h)
        attention_output_v = self.attn(query_v, key_v, value_v)
        attention_output_h = attention_output_h.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()
        attention_output_v = attention_output_v.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()
        attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)


        return x

class AMM(nn.Module):
    def __init__(self):
        super(AMM, self).__init__()

        head_num = 16
        dim = 64
        self.conv_1 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_1 = Intra_SA(dim, head_num)
        self.Trans_block_2 = Inter_SA(dim, head_num)
        
        self.conv_2 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_3 = Intra_SA(dim, head_num)
        self.Trans_block_4 = Inter_SA(dim, head_num)
        
        self.conv_3 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_5 = Intra_SA(dim, head_num)
        self.Trans_block_6 = Inter_SA(dim, head_num)
        
        self.conv_4 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_7 = Intra_SA(dim, head_num)
        self.Trans_block_8 = Inter_SA(dim, head_num)
        
        self.conv_5 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_9 = Intra_SA(dim, head_num)
        self.Trans_block_10 = Inter_SA(dim, head_num)
        
        self.conv_6 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.Trans_block_11 = Intra_SA(dim, head_num)
        self.Trans_block_12 = Inter_SA(dim, head_num)

    def forward(self, x, prior):

        hx = torch.cat([x, prior], dim=1)
        hx = self.conv_1(hx)
        hx = self.Trans_block_1(hx)                   
        hx = self.Trans_block_2(hx)
        
        hx = torch.cat([hx, prior], dim=1)
        hx = self.conv_2(hx)
        hx = self.Trans_block_3(hx)
        hx = self.Trans_block_4(hx)
        
        hx = torch.cat([hx, prior], dim=1)
        hx = self.conv_3(hx)
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)
        
        hx = torch.cat([hx, prior], dim=1)
        hx = self.conv_4(hx)
        hx = self.Trans_block_7(hx)
        hx = self.Trans_block_8(hx)
        
        hx = torch.cat([hx, prior], dim=1)
        hx = self.conv_5(hx)
        hx = self.Trans_block_9(hx)
        hx = self.Trans_block_10(hx)
        
        hx = torch.cat([hx, prior], dim=1)
        hx = self.conv_6(hx)
        hx = self.Trans_block_11(hx)
        hx = self.Trans_block_12(hx)
    

        return hx + x




