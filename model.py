
from config import ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerLayerNorm
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

#Position embeddings
class PositionEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims = ModelArgs.embeddings_dims,
        block_size = ModelArgs.block_size
    ):
        super().__init__()

        self.position_embeddings = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=ModelArgs.device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        # nn.init.normal_(self.position_embeddings.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.position_embeddings




# Text embeddings
class TgtTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size = ModelArgs.tgt_vocab_size,
        embeddings_dims = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings = ModelArgs.tgt_vocab_size, embedding_dim=embeddings_dims, device=ModelArgs.device) #Just a look up table to convert the toekns_ids to some numbers
        # nn.init.normal_(self.embeddings_table.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.embeddings_table(x)




#Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = ModelArgs.embeddings_dims
    ):
        super().__init__()
        if(ModelArgs.use_liger == False):
            self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
        else:
            self.norm = LigerLayerNorm(embeddings_dims)

    def forward(self, x):

        return self.norm(x)





#FeedForward Neural Network

class MLPBlock(nn.Module):
    def __init__(
        self,
        dropout = ModelArgs.dropout,
        embeddings_size = ModelArgs.embeddings_dims,
        # inner_dimensional_states: int = 3072
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(device=ModelArgs.device, in_features=embeddings_size, out_features= 4 * ModelArgs.embeddings_dims),
            nn.GELU(),
            nn.Linear(device=ModelArgs.device, in_features= 4 * ModelArgs.embeddings_dims, out_features=embeddings_size),
            nn.Dropout(p = dropout)
        )

    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        return self.mlp(x)




class MaskedAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        if(ModelArgs.use_flash_attention==False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=ModelArgs.device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)

        if(ModelArgs.use_flash_attention):
            # Combined linear projections for Q, K, V
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=ModelArgs.device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)

    def forward(self, x):
        # print(x.shape)
        batch, block_size, embd_dims = x.shape
        if(ModelArgs.use_flash_attention == False):
            k = self.keys(x)
            q = self.query(x)
            v = self.values(x)
        # if(use_flash_attention == False):
            masked_table = torch.tril(torch.ones(block_size, block_size, device=ModelArgs.device))
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=ModelArgs.dropout, is_causal=True
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, block_size, -1)
            return out

            


class MaskedMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out


        
#Single Attention Head

class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.head_size = embeddings_dims // no_of_heads
        if(ModelArgs.use_flash_attention == False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=ModelArgs.device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)

        # if(use_flash_attention):
        #     # Combined linear projections for Q, K, V
        #     self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)

    def forward(self, query, key, value, mask=None):


        batch, block_size, embd_dims = query.shape
        if(ModelArgs.use_flash_attention == False):
            q = self.query(query)
            k = self.keys(key)
            v = self.values(value)


        if(ModelArgs.use_flash_attention):

            batch, q_seq_len, _ = query.shape
            _, k_seq_len, _ = key.shape
            _, v_seq_len, _ = value.shape
            q = query.view(batch, q_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            k = key.view(batch, k_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            v = value.view(batch, v_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=ModelArgs.dropout, is_causal=False
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, q_seq_len, -1)
            return out
        else:   
            masked_table = torch.tril(torch.ones(block_size, block_size, device=ModelArgs.device))
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out


        #Single Attention Head

class FullAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.head_size = embeddings_dims // no_of_heads
        if(ModelArgs.use_flash_attention == False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=ModelArgs.device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)
        if(ModelArgs.use_flash_attention):
            # Combined linear projections for Q, K, V
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=ModelArgs.device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)


    def forward(self, x, mask=None):
        batch, block_size, embd_dims = x.shape
        if(ModelArgs.use_flash_attention == False):
            k = self.keys(x)
            q = self.query(x)
            v = self.values(x)
        # masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
        if(ModelArgs.use_flash_attention):
  
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=ModelArgs.dropout, is_causal=False
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, block_size, -1)
            return out

        else:
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            if(mask != None):
                mask = mask.unsqueeze(1)
                masked_values = weights.masked_fill(mask == 0, float('-inf'))
                weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
                # weights_normalized = self.dropout(weights_normalized)
                out = weights_normalized @ v
                out = self.dropout(out)
                return out
            else:
                weights_normalized = nn.functional.softmax(weights, dim=-1) #Normalize along the embeddings dimension for all the tokens
                # weights_normalized = self.dropout(weights_normalized)
                out = weights_normalized @ v
                out = self.dropout(out)
                return out


            
class   FullMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features= self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out



        

class CrossMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)

    def forward(self, value, key, x, mask=None):
        concat = torch.cat([head(x, key, value,  mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out



    # Decoder Block

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        dropout = ModelArgs.dropout,
        # vocab_size = vocab_size
    ):
        super().__init__()

        self.cross = CrossMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.masked = MaskedMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        # self.layer_norm3 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.layer_norm4 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, key, value, x, mask=None):
        masked_out = self.masked(x)
        cross_out = self.cross(value, key, x, mask)
        x = x + self.layer_norm1(masked_out) #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        # print(x.shape)
        x = x + self.layer_norm2(cross_out) #Very important step
        # print(x.shape)
        # x = x + self.mha(self.layer_norm1(x))  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + self.layer_norm4(self.mlp_block(x)) #Very important step
        # print(x.shape)

        return x



        # Decoder Block

class DecoderModel(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        block_size = ModelArgs.block_size,
        dropout = ModelArgs.dropout,
        no_of_decoder_layers = ModelArgs.no_of_decoder_layers,
        # vocab_size = vocab_size
    ):
        super().__init__()




        # self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        # self.positional_embeddings_tgt = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        self.positional_embeddings_tgt = PositionEmbeddings()
        # torch.nn.init.normal_(self.positional_embeddings_tgt, mean=0.0, std=0.02)

        # out = self.decoder_layers(query, key, x)
        # Loop through each decoder layer
    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, key, value, x, mask):
        # x = self.tgt_text_embds(x)
        x = x + self.positional_embeddings_tgt(x)[:, :x.shape[1], :]
        # print(x.shape)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(key, value, x, mask)
        x = self.layer_norm(x)

        return x





class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        dropout = ModelArgs.dropout,
        mask=None
    ):
        super().__init__()

        self.mha = FullMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, mask=None):
        # print(self.mha(x, mask).shape)
        # print(x.shape)
        mha_out = self.mha(x, mask)
        # mha_out = mha_out
        # print(mha_out.shape)
        x = x + self.layer_norm1(mha_out)
        x = x + self.layer_norm2(self.mlp_block(x))

        return x


        


class EncoderModel(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        block_size = ModelArgs.block_size,
        dropout = ModelArgs.dropout,
        no_of_decoder_layers = ModelArgs.no_of_decoder_layers,
        # vocab_size = vocab_size
    ):
        super().__init__()


        # self.positional_embeddings_src = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size

        self.conv1 = nn.Conv1d(in_channels=ModelArgs.n_channels, out_channels=embeddings_dims, kernel_size=ModelArgs.kernel_size, device=ModelArgs.device, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embeddings_dims, out_channels=embeddings_dims, kernel_size=ModelArgs.kernel_size, device=ModelArgs.device, padding=1)

        self.positional_embeddings_src = PositionEmbeddings()

        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)

    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):

        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        # print("Shape: ", x.shape)
        # x = self.src_text_embeds(x)
        # print(self.positional_embeddings_src.shape)
        x = x.permute(0, 2, 1)
        # print("Shape: ", x.shape)
        # print(self.positional_embeddings_src(x).shape)
        x = x + self.positional_embeddings_src(x)
        # print(x)
        # print(x.shape)
        # Loop through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x



        

class Whisper(nn.Module):
    def __init__(
        self,

    ):
        super().__init__()

        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        # self.pos = PositionalEmbeddings()
        self.tgt_text_embds = TgtTextEmbeddings(vocab_size=ModelArgs.tgt_vocab_size, embeddings_dims=ModelArgs.embeddings_dims)
        self.linear_layer = nn.Linear(in_features=ModelArgs.embeddings_dims, out_features=ModelArgs.tgt_vocab_size, device=ModelArgs.device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.le_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        ).to(ModelArgs.device)
        # self.src_text_embeds = SrcTextEmbeddings(vocab_size=src_vocab_size, embeddings_dims=embeddings_dims)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, actual_labels=None, inference=False):
        # print("Here: ", src.shape)
        # print("Here2: ", tgt.shape)
        # x = self.src_text_embeds(src)
        x = self.encoder(src, src_mask)

        y = self.tgt_text_embds(tgt)
        # print(x.shape)
        y = self.decoder(x, x, y, tgt_mask)
        # print(y.shape)
        if(inference):
            out = self.linear_layer(y)
            return out
        if(ModelArgs.use_liger):  
            y = y.contiguous().view(-1, ModelArgs.embeddings_dims)
            labels = actual_labels.contiguous().view(-1)
            
            # Pass linear layer weights FIRST as required [2][5]
            loss = self.le_loss(self.linear_layer.weight, y, labels)
            return loss
        else:
            out = self.linear_layer(y)
            return out
