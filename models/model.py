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
                       intermidiate_size,
                       itermidiate_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       normalization_type,
                       dropout_prob):
        super().__init__()
        self.n_layers = n_layers
        args = [model_size,
                intermidiate_size,
                itermidiate_kernel_size,
                activation,
                n_heads,
                size_per_head,
                normalization_type,
                dropout_prob]
        self.layers = nn.ModuleList([TransformerEncoderLayer(*args) for i in range(n_layers)])

    def forward(self, x, attention_mask):
        for i in range(self.n_layers):
            x, attention_mask = self.layers[i](x, attention_mask)
        return x


class SinCosPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size,
                 maxlen):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embeddings):
        return embeddings + self.pos_embedding[:embeddings.shape[0], :]


def duplicate_by_duration(encoder_result, durations, device):
    bs = encoder_result.shape[0]
    results = []
    for i in range(bs):
        melspec_len = durations[i].sum()
        durations_cumsum = durations[i].cumsum(0)
        mask1 = torch.arange(melspec_len)[None, :].to(device) < (durations_cumsum[:, None])
        mask2 = torch.arange(melspec_len)[None, :].to(device) >= (durations_cumsum - durations[i])[:, None]
        mask = (mask2 * mask1).float()
        results.append(mask.T @ encoder_result[i])
    results = pad_sequence(results).permute(1, 0, 2)
    return results


class FastSpeechModel(nn.Module):
    def __init__(self, vocab_size,
                       max_len,
                       n_layers,
                       output_size,
                       model_size,
                       intermidiate_size,
                       itermidiate_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       normalization_type,
                       dropout_prob,
                       device):
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'gelu':
            activation = nn.GELU
        args = [n_layers,
                model_size,
                intermidiate_size,
                itermidiate_kernel_size,
                activation,
                n_heads,
                size_per_head,
                normalization_type,
                dropout_prob]
        self.tokens_positions = SinCosPositionalEncoding(model_size, max_len)
        self.frames_positions = SinCosPositionalEncoding(model_size, max_len)
        self.embedding_layer = nn.Embedding(vocab_size, model_size)
        self.encoder = Transformer(*args)
        self.decoder = Transformer(*args)
        self.output_layer = nn.Linear(model_size, output_size)
        self.duration_predictor = DurationPredictor(model_size)
        self.device = device

    def forward(self, batch, train=True):
        tokens = batch["tokens"]
        tokens_length = batch["token_lengths"]

        tokens_embeddings = self.embedding_layer(tokens)
        tokens_embeddings = tokens_embeddings + self.tokens_positions(tokens_embeddings)

        attention_mask = (torch.arange(tokens.shape[1])[None, :].to(self.device) > tokens_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        encoder_result = self.encoder(tokens_embeddings, attention_mask)
        length_predictions = self.duration_predictor(encoder_result).squeeze(2)

        if train:
            melspec_length = batch["melspec_length"]
            duration_multipliers = batch["duration_multipliers"]
            input_to_decoder = duplicate_by_duration(encoder_result, duration_multipliers, self.device)
            mask = (torch.arange(input_to_decoder.shape[1])[None, :].to(self.device) <= melspec_length[:, None]).float()
            input_to_decoder = input_to_decoder * mask[:, :, None]
            attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :].to(self.device) > melspec_length[:, None]).float()
            attention_mask[attention_mask == 1] = -torch.inf
        else:
            duration_multipliers = (torch.exp(length_predictions) - 1).round().int()
            duration_multipliers[duration_multipliers < 1] = 1
            input_to_decoder = duplicate_by_duration(encoder_result, duration_multipliers, self.device)
            attention_mask = torch.zeros(input_to_decoder.shape[:2]).to(self.device)
        output = self.decoder(input_to_decoder, attention_mask)
        output = self.output_layer(output)
        output = output.permute(0, 2, 1)
        return output, length_predictions
    

class SelfAttention(nn.Module):
    def __init__(self, input_size, n_heads, size_per_head):
        super().__init__()
        self.n_heads = n_heads
        self.size_per_head = size_per_head
        self.queries = nn.Linear(input_size, n_heads * size_per_head)
        self.keys = nn.Linear(input_size, n_heads * size_per_head)
        self.values = nn.Linear(input_size, n_heads * size_per_head)

    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        bs, seq_len = x.shape[:2]
        x_reshaped = x.reshape(-1, x.shape[2])
        queries = self.queries(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)
        keys = self.keys(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)
        values = self.values(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)

        queries_scaled = queries * 1 / math.sqrt(self.size_per_head)
        before_softmax = torch.einsum('bsij, bskj->bsik', queries_scaled, keys)
        after_softmax = torch.softmax(before_softmax + attention_mask.unsqueeze(1).unsqueeze(1), dim=-1)
        context = torch.einsum('bsik, bskj->bsij', after_softmax, values)
        return context.permute(0, 2, 1, 3).flatten(2)


class IntermidiateLayer(nn.Module):
    def __init__(self, model_size, intermidiate_size, kernel_size, activation, normalization_type, dropout_prob):
        super().__init__()
        self.intermidiate = nn.Sequential(nn.Conv1d(model_size, intermidiate_size, kernel_size=kernel_size[0], padding='same'),
                                          activation(),
                                          nn.Conv1d(intermidiate_size, model_size, kernel_size=kernel_size[1], padding='same'))
        self.layer_norm = nn.LayerNorm(model_size)
        self.normalization_type = normalization_type
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        if self.normalization_type == 'pre':
            to_add = self.layer_norm(x)
            to_add = to_add.permute(0, 2, 1)
            to_add = self.dropout(self.intermidiate(to_add)).permute(0, 2, 1)
            x = x + to_add
        elif self.normalization_type == 'post':
            x = x.permute(0, 2, 1)
            to_add = self.dropout(self.intermidiate(x))
            x = (x + to_add).permute(0, 2, 1)
            x = self.layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_size,
                 intermidiate_size,
                 itermidiate_kernel_size,
                 activation,
                 n_heads,
                 size_per_head,
                 normalization_type,
                 dropout_prob):
        super().__init__()
        self.self_attention = SelfAttention(model_size, n_heads, size_per_head)
        self.context_linear = nn.Linear(n_heads * size_per_head, model_size)
        self.intermidiate_size = intermidiate_size
        self.intermediate_layer = IntermidiateLayer(model_size,
                                                    intermidiate_size,
                                                    itermidiate_kernel_size,
                                                    activation,
                                                    normalization_type,
                                                    dropout_prob)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(model_size)
        self.normalization_type = normalization_type
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        attention_output = self.self_attention(x, attention_mask)
        if self.normalization_type == 'pre':
            to_add = self.dropout(self.context_linear(self.layer_norm(attention_output)))
            x = x + to_add
        elif self.normalization_type == 'post':
            to_add = self.dropout(self.context_linear(attention_output))
            x = self.layer_norm(x + to_add)
        x = self.intermediate_layer(x)
        return x, attention_mask