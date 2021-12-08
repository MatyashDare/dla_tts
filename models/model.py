import torch
from torch import nn
import math
from torch.nn.utils.rnn import pad_sequence

class ConvPermute(nn.Module):
    def __init__(self, embedding_size, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(embedding_size, embedding_size, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1)
        return y


class DurationPredictor(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.model = nn.Sequential(
            ConvPermute(embedding_size, kernel_size=3, padding='same'),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            ConvPermute(embedding_size, kernel_size=3, padding='same'),
            nn.LayerNorm(embedding_size),
            nn.ReLU()
        )
        self.output = nn.Linear(embedding_size, 1)

    def forward(self, x):
        y = self.model(x)
        return self.output(y)

    
    



class Transformer(nn.Module):
    def __init__(self, n_layers,
                       model_size,
                       inter_size,
                       iter_kernel_size,
                       heads_num,
                       size_head,
                       p):
        super().__init__()
        self.n_layers = n_layers
        args = [model_size, inter_size, iter_kernel_size,heads_num, size_head, p]
        self.layers = nn.ModuleList([Encoder(*args) for i in range(n_layers)])

    def forward(self, x, attention_mask):
        for i in range(self.n_layers):
            x, attention_mask = self.layers[i](x, attention_mask)
        return x

####
class SinCosPE(nn.Module):
    def __init__(self,
                 emb_size,
                 max_len):
        super().__init__()
        
        pe = torch.zeros((max_len, emb_size))
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        d = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * d), torch.cos(pos * d)
        # change dims
        pe = pe.unsqueeze(-2)
        # add persistent buffer to the module
        self.register_buffer('pos_embedding', pe)

    def forward(self, embeds):
        return embeds + self.pos_embedding[:embeds.shape[0], :]
    

def duplication_alignment(encoder_result, durations, device):
    bs = encoder_result.shape[0]
    ans = []
    for i in range(bs):
        # суммирование элементов массива с накоплением
        durations_cumsum = durations[i].cumsum(0)
        mask1 = torch.arange(durations[i].sum())[None, :].to(device) < (durations_cumsum[:, None])
        mask2 = torch.arange(durations[i].sum())[None, :].to(device) >= (durations_cumsum - durations[i])[:, None]
        mask12 = (mask2 * mask1).float()
        to_add = mask12.T @ encoder_result[i]
        ans.append(to_add)
    return pad_sequence(ans).permute(1, 0, 2)

# copy-paste + length predictions
# https://github.com/xcmyz/FastSpeech
# --- FastSpeechModel ---
class FastSpeechModel(nn.Module):
    def __init__(self, vocab_size,
                       max_len,
                       n_layers,
                       output_size,
                       model_size,
                       inter_size,
                       inter_kernel_size,
                       heads_num,
                       size_head,
                       p,
                       device):
        super().__init__()

        activation = nn.ReLU
        args = [n_layers,
                model_size,
                inter_size,
                inter_kernel_size,
                activation,
                heads_num,
                size_head,
                p]
        self.tokens_positions = SinCosPE(model_size, max_len)
        self.frames_positions = SinCosPE(model_size, max_len)
        self.embedding_layer = nn.Embedding(vocab_size, model_size)
        self.encoder = Transformer(*args)
        self.decoder = Transformer(*args)
        self.output_layer = nn.Linear(model_size, output_size)
        self.duration_predictor = DurationPredictor(model_size)
        self.device = device
####
    def forward(self, batch, train=True):
        tokens, tokens_length = batch["tokens"], batch["token_lengths"]
        tokens_embeds_to_add = self.embedding_layer(tokens)
        tokens_embeds = tokens_embeds_to_add + self.tokens_positions(tokens_embeds_to_add)
        attention_mask = (torch.arange(tokens.shape[1])[None, :].to(self.device) > tokens_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        encoder_ans = self.encoder(tokens_embeds, attention_mask)
        length_predictions = self.duration_predictor(encoder_ans).squeeze(2)
        if not train:
            dm = (torch.exp(length_predictions) - 1).round().int()
            dm[dm < 1] = 1
            to_decoder = duplication_alignment(encoder_ans, duration_multipliers, self.device)
            attention_mask = torch.zeros(input_to_decoder.shape[:2]).to(self.device) 
        else:
            melspec_length, dm = batch["melspec_length"], batch["duration_multipliers"]
            to_decoder = duplication_alignment(encoder_ans, dm, self.device)
            mask = (torch.arange(to_decoder.shape[1])[None, :].to(self.device) <= melspec_length[:, None]).float()
            to_decoder = to_decoder * mask[:, :, None]
            attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :].to(self.device) > melspec_length[:, None]).float()
            attention_mask[attention_mask == 1] = -torch.inf
        output = self.decoder(to_decoder, attention_mask)
        output = self.output_layer(output)
        output = output.permute(0, 2, 1)
        return output, length_predictions

    

class SelfAttention(nn.Module):
    def __init__(self, input_size, heads_num, size_head):
        super().__init__()
        self.size_head = size_head
        self.heads_num = heads_num
        self.Q = nn.Linear(input_size, size_head * heads_num)
        self.K = nn.Linear(input_size, size_head * heads_num)
        self.V = nn.Linear(input_size, size_head * heads_num)
# Attention
    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        bs = x.shape[0]
        seq_len = x.shape[1]
        #change shapes
        x_reshaped = x.reshape(-1, x.shape[2])
        Q = self.Q(x_reshaped).reshape(bs, seq_len, self.heads_num, self.size_head)
        Q = Q.permute(0, 2, 1, 3)
        K = self.K(x_reshaped).reshape(bs, seq_len, self.heads_num, self.size_head)
        K = K.permute(0, 2, 1, 3)
        V = self.V(x_reshaped).reshape(bs, seq_len, self.heads_num, self.size_head)
        V = V.permute(0, 2, 1, 3)
        #scale Q-matrix
        scaledQ = Q * 1 / math.sqrt(self.size_head)
        # Einstein summation
        sum_to_softmax = torch.einsum('bsij, bskj->bsik', scaledQ, K)
        #add empty dimensions
        to_add = attention_mask.unsqueeze(1).unsqueeze(1)
        #apply Softmax()
        to_softmax = torch.softmax(sum_to_softmax + to_add, dim=-1)
        # Einstein summation Softmax(scaled Q-matrix) and V-matrix
        att_context = torch.einsum('bsik, bskj->bsij', to_softmax, V)
        att_context = att_context.permute(0, 2, 1, 3)
        return att_context.flatten(2)

class InterLayer(nn.Module):
    def __init__(self, model_size, inter_size, kernel_size, p):
        super().__init__()

        #default activation nn.ReLU()
        self.inter = nn.Sequential(nn.Conv1d(model_size, inter_size, kernel_size=kernel_size, padding='same'),
                                          nn.ReLU(),
                                          nn.Conv1d(inter_size, model_size, kernel_size=1, padding='same'))
        self.layer_norm = nn.LayerNorm(model_size)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        to_add = self.dropout(self.inter(x))
        x = (x + to_add).permute(0, 2, 1)
        x = self.layer_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, model_size,
                 inter_size,
                 iter_kernel_size,
                 heads_num,
                 size_head,
                 p):
        super().__init__()
        self.self_attention = SelfAttention(model_size, heads_num, size_head)
        self.context_linear = nn.Linear(heads_num * size_head, model_size)
        self.inter_size = inter_size
        self.inter_layer = InterLayer(model_size, inter_size, iter_kernel_size, p)
        self.layer_norm = nn.LayerNorm(model_size)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x, attention_mask):
        attention_out = self.self_attention(x, attention_mask)
        x = self.layer_norm(x + self.dropout(self.context_linear(attention_out)))
        return self.inter_layer(x), attention_mask

