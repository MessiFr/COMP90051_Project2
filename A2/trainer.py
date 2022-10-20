import imp
import torch
from model import modeler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import sys
import os
import time
from dataset import Dataset
from evaluate import Evaluator
class Trainer():
    def __init__(self):
        self.input_data = Dataset(data_file="D:/jupyter/sophia/train.json")
        self.embed_d = 128
        self.word_n = self.input_data.word_n
        self.word_dim = 300
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c_len = 100
        self.iter_max = 500
        self.batch_s = 512
        self.lr = 0.001
        self.margin_d = 0.1
        self.c_tradeoff = 0.1
        self.c_reg = 0.001
        self.embedder = "camel"
        self.model_path = "./saved_model"
        self.metric = "dot"
        self.early_stop = 20
        self.top_K = [1,2,5,10,20]
        self.save = "True"
        self.evaluator = Evaluator(self.input_data, self.metric, self.early_stop, self.top_K, self.save)
    def pad(self, list_of_tensors):
        list_of_tensors = [torch.LongTensor(elem[:self.c_len]) for elem in list_of_tensors]
        seq_lengths = torch.LongTensor([len(elem) for elem in list_of_tensors])
        seq_tensor = pad_sequence(list_of_tensors, batch_first=True)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, perm_idx, seq_lengths

    def training(self):
        model = modeler(self.input_data, self.embed_d, self.word_n, self.word_dim, self.device).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)

        for epoch in range(1, self.iter_max):
            totalLoss = 0
            print(epoch)
            p_a_a_dir_batch = self.input_data.p_a_a_dir().to(self.device)
            p_c_dir_batch = self.input_data.get_content(p_a_a_dir_batch)
            p_c_dir_batch, perm_idx_dir, seq_lengths_dir = self.pad(p_c_dir_batch)
            p_a_a_dir_batch = p_a_a_dir_batch[perm_idx_dir]
            p_c_dir_batch = p_c_dir_batch.to(self.device)

            p_a_a_indir_batch = self.input_data.p_a_a_indir().to(self.device)
            p_c_indir_batch = self.input_data.get_content(p_a_a_indir_batch)
            p_c_indir_batch, perm_idx_indir, seq_lengths_indir = self.pad(p_c_indir_batch)
            p_a_a_indir_batch = p_a_a_indir_batch[perm_idx_indir]
            p_c_indir_batch = p_c_indir_batch.to(self.device)
            mini_batch_n = int(len(p_a_a_dir_batch) / self.batch_s)
            for i in range(mini_batch_n):
                p_a_a_dir_mini_batch = p_a_a_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                p_c_dir_mini_batch = p_c_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                seq_lengths_dir_mini_batch = seq_lengths_dir[i * self.batch_s:(i + 1) * self.batch_s]

                p_a_a_indir_mini_batch = p_a_a_indir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                p_c_indir_mini_batch = p_c_indir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                seq_lengths_indir_mini_batch = seq_lengths_indir[i * self.batch_s:(i + 1) * self.batch_s]

                optimizer.zero_grad()

                pos_dir, neg_dir, sum1, sum2 = model(p_a_a_dir_mini_batch, p_a_a_indir_mini_batch, p_c_dir_mini_batch,
                                                     p_c_indir_mini_batch, seq_lengths_dir_mini_batch,
                                                     seq_lengths_indir_mini_batch, isTrain=True)

                Loss_1 = torch.sum(torch.max(pos_dir - neg_dir + self.margin_d, torch.Tensor([0]).to(self.device)))
                Loss_2 = (-(sum1 + sum2)).sum()

                reg_loss = None
                for param in list(filter(lambda p: p.requires_grad, model.parameters())):
                    if reg_loss is None:
                        reg_loss = (param ** 2).sum()
                    else:
                        reg_loss = reg_loss + (param ** 2).sum()

                loss = Loss_1 + self.c_tradeoff * Loss_2 + self.c_reg * reg_loss

                loss.backward()
                optimizer.step()

                totalLoss += loss.item()
                del loss


            st = "[{}][Iter {:3}] loss: {:3}".format(self.currentTime(), epoch, round(totalLoss,2))
            model.eval()
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # torch.save(model.state_dict(), self.model_path + "/camel_epo{}_loss{}.pth".format(epoch, str(round(totalLoss,2))))
            p_text_all = [self.input_data.context_data[i] for i in self.input_data.test_idx]

            p_text_all, perm_idx, seq_lengths = self.pad(p_text_all)
            p_text_deep_f = model([], [], p_text_all, [], seq_lengths, [], isTrain=False)
            perm_idx = perm_idx.numpy().argsort()
            p_text_deep_f = p_text_deep_f[perm_idx]

            p_text_deep_f = p_text_deep_f.detach().cpu().numpy()
            a_latent_f = model.author_embed.weight.data.detach().cpu().numpy()
            # print(st)
            is_converged = self.evaluator.evaluate_Camel(model, self.embedder, self.model_path, st, epoch, p_text_deep_f, a_latent_f)
            if is_converged:
                print("Converged!")
                return
            model.train()
    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s