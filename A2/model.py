from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch


class modeler(nn.Module):
    def __init__(self, input_data, embed_d, word_n, word_dim, device):
        super(modeler, self).__init__()
        self.author_embed = nn.Embedding(input_data.author_num, embed_d)
        self.word_embed = nn.Embedding(word_n + 2, word_dim)
        self.word_embed.weight.requires_grad = False
        self.rnn_words = nn.GRU(word_dim, embed_d, batch_first=True)
        self.bias = nn.Parameter(torch.zeros((1,1)))
        self.logsigmoid = nn.LogSigmoid()
        # self.init_weights(input_data.word_embed)

        self.embed_d = embed_d
        self.input_data = input_data
        self.device = device

    def init_weights(self, word_embed):
        nn.init.normal_(self.author_embed.weight.data, mean=0.0, std=0.01)
        self.bias.data.fill_(0.1)
        # self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

    def forward(self, p_a_a_dir, p_a_a_indir, p_c_dir_input, p_c_indir_input, seq_lengths_dir, seq_lengths_indir, isTrain=True):
        if isTrain:
            # Metric learning loss
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
            packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
            p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float().to(self.device)

        if not isTrain:
            # Metric learning loss
            self.word_embed = self.word_embed.cpu()
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
            packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)
            self.rnn_words = self.rnn_words.cpu()
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
            p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float()
            self.word_embed = self.word_embed.to(self.device)
            self.rnn_words = self.rnn_words.to(self.device)
            return p_c_dir_e

        a_e_pos = self.author_embed(p_a_a_dir[:, 1])
        a_e_neg = self.author_embed(p_a_a_dir[:, 2])

        pos_dir = torch.sum((p_c_dir_e - a_e_pos) ** 2, 1)
        neg_dir = torch.sum((p_c_dir_e - a_e_neg) ** 2, 1)

        # Random walk loss
        p_c_indir_word_e = self.word_embed(p_c_indir_input)
        packed_input = pack_padded_sequence(p_c_indir_word_e, seq_lengths_indir.cpu().numpy(), batch_first=True)
        p_c_indir_deep_e, _ = self.rnn_words(packed_input)
        p_c_indir_deep_e, _ = pad_packed_sequence(p_c_indir_deep_e, batch_first=True)

        p_c_indir_e = torch.sum(p_c_indir_deep_e, 1) / seq_lengths_indir.unsqueeze(1).float().to(self.device)
        a_e_pos = self.author_embed(p_a_a_indir[:, 1])
        a_e_neg = self.author_embed(p_a_a_indir[:, 2])

        sum1 = torch.sum(p_c_indir_e * a_e_pos, 1) + self.bias
        sum2 = torch.sum(p_c_indir_e * a_e_neg, 1) + self.bias
        sum1 = self.logsigmoid(sum1)
        sum2 = self.logsigmoid(-sum2)

        return pos_dir, neg_dir, sum1, sum2