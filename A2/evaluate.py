import re
import pdb
import sys
import torch
from numpy import linalg as LA
import numpy as np
import os
import pickle as pkl

class Evaluator:
    def __init__(self, dataset, metric, early_stop, top_Ks, save):
        self.dataset = dataset
        self.metric = metric
        self.early_stop = early_stop
        self.top_Ks = top_Ks
        self.save = save
        self.file_name = "_{}".format("test101")
        p_id_map = dict()
        new_id_temp = 0
        for p_id_temp in self.dataset.test_p_a_dir:
            p_id_map[p_id_temp] = new_id_temp
            new_id_temp += 1

        p_a_idx_neg_dict_test = dict()
        # a_idx_p_neg_dict_test = dict()
        # p_a_neg_ids_f = open(self.dataset.data_path + "/paper_author_neg_ids.txt", "r")


        self.p_id_map = p_id_map
        self.p_a_idx_neg_dict_test = dataset.p_a_neg_dir_test()
        # self.a_idx_p_neg_dict_test = a_idx_p_neg_dict_test

        self.recall_ave_dev_dict = {}; self.pre_ave_dev_dict = {}
        self.AUC_ave_dev_dict = {}; self.evaluate_p_num_dev_dict = {}

        self.recall_ave_test_dict = {}; self.pre_ave_test_dict = {}
        self.AUC_ave_test_dict = {}; self.evaluate_p_num_test_dict = {}

        self.recall_ave_dev_dict_list = {}; self.pre_ave_dev_dict_list = {}
        self.AUC_ave_dev_dict_list = {}; self.F1_ave_dev_dict_list = {}

        self.recall_ave_test_dict_list = {}; self.pre_ave_test_dict_list = {}
        self.AUC_ave_test_dict_list = {}; self.F1_ave_test_dict_list = {}

        self.recall_ave_dev_dict_list_best = {}; self.pre_ave_dev_dict_list_best = {}
        self.AUC_ave_dev_dict_list_best = {}; self.F1_ave_dev_dict_list_best = {}

        self.recall_ave_test_dict_list_best = {}; self.pre_ave_test_dict_list_best = {}
        self.AUC_ave_test_dict_list_best = {}; self.F1_ave_test_dict_list_best = {}

        self.recall_ave_dev_dict_best = {}; self.pre_ave_dev_dict_best = {}
        self.AUC_ave_dev_dict_best = {}; self.F1_ave_dev_dict_best = {}

        self.recall_ave_test_dict_best = {}; self.pre_ave_test_dict_best = {}
        self.AUC_ave_test_dict_best = {}; self.F1_ave_test_dict_best = {}

        for top_K in top_Ks:
            self.recall_ave_dev_dict.setdefault(top_K, 0); self.pre_ave_dev_dict.setdefault(top_K, 0)
            self.AUC_ave_dev_dict.setdefault(top_K, 0); self.evaluate_p_num_dev_dict.setdefault(top_K, 0)

            self.recall_ave_test_dict.setdefault(top_K, 0); self.pre_ave_test_dict.setdefault(top_K, 0)
            self.AUC_ave_test_dict.setdefault(top_K, 0); self.evaluate_p_num_test_dict.setdefault(top_K, 0)

            self.recall_ave_dev_dict_list.setdefault(top_K, []); self.pre_ave_dev_dict_list.setdefault(top_K, [])
            self.AUC_ave_dev_dict_list.setdefault(top_K, []); self.F1_ave_dev_dict_list.setdefault(top_K, [])

            self.recall_ave_test_dict_list.setdefault(top_K, []); self.pre_ave_test_dict_list.setdefault(top_K, [])
            self.AUC_ave_test_dict_list.setdefault(top_K, []); self.F1_ave_test_dict_list.setdefault(top_K, [])

            self.recall_ave_dev_dict_list_best.setdefault(top_K, []); self.pre_ave_dev_dict_list_best.setdefault(top_K, [])
            self.AUC_ave_dev_dict_list_best.setdefault(top_K, []); self.F1_ave_dev_dict_list_best.setdefault(top_K, [])

            self.recall_ave_test_dict_list_best.setdefault(top_K, []); self.pre_ave_test_dict_list_best.setdefault(top_K, [])
            self.AUC_ave_test_dict_list_best.setdefault(top_K, []); self.F1_ave_test_dict_list_best.setdefault(top_K, [])

            self.recall_ave_dev_dict_best.setdefault(top_K, 0); self.pre_ave_dev_dict_best.setdefault(top_K, 0)
            self.AUC_ave_dev_dict_best.setdefault(top_K, 0); self.F1_ave_dev_dict_best.setdefault(top_K, 0)

            self.recall_ave_test_dict_best.setdefault(top_K, 0); self.pre_ave_test_dict_best.setdefault(top_K, 0)
            self.AUC_ave_test_dict_best.setdefault(top_K, 0); self.F1_ave_test_dict_best.setdefault(top_K, 0)


    def compute_score(self, x, y, metric='L2'):
        if metric == 'L2':
            return -LA.norm(x-y)
        elif metric == 'dot':
            return np.dot(x,y)

    def print_result(self, model, embedder, model_path, epoch, st):
        recall_st_dev = ''; precision_st_dev = ''
        f1_st_dev = ''; AUC_st_dev = ''

        recall_st_test = ''; precision_st_test = ''
        f1_st_test = ''; AUC_st_test = ''

        recall_st_dev_best = ''; precision_st_dev_best = ''
        f1_st_dev_best = ''; AUC_st_dev_best = ''

        recall_st_test_best = ''; precision_st_test_best = ''
        f1_st_test_best = ''; AUC_st_test_best = ''

        for idx, top_K in enumerate(self.top_Ks):
            recall_ave_dev = self.recall_ave_dev_dict[top_K] / self.evaluate_p_num_dev_dict[top_K]
            pre_ave_dev = self.pre_ave_dev_dict[top_K] / self.evaluate_p_num_dev_dict[top_K]
            AUC_ave_dev = self.AUC_ave_dev_dict[top_K] / self.evaluate_p_num_dev_dict[top_K]
            F1_ave_dev = (2 * recall_ave_dev * pre_ave_dev) / (recall_ave_dev + pre_ave_dev)

            recall_ave_test = self.recall_ave_test_dict[top_K] / self.evaluate_p_num_test_dict[top_K]
            pre_ave_test = self.pre_ave_test_dict[top_K] / self.evaluate_p_num_test_dict[top_K]
            AUC_ave_test = self.AUC_ave_test_dict[top_K] / self.evaluate_p_num_test_dict[top_K]
            F1_ave_test = (2 * recall_ave_test * pre_ave_test) / (recall_ave_test + pre_ave_test)

            recall_st_dev += '{},'.format(round(recall_ave_dev,4))
            precision_st_dev += '{},'.format(round(pre_ave_dev,4))
            if idx == 0:
                AUC_st_dev += '{},'.format(round(AUC_ave_dev,4))
            f1_st_dev += '{},'.format(round(F1_ave_dev,4))

            recall_st_test += '{},'.format(round(recall_ave_test,4))
            precision_st_test += '{},'.format(round(pre_ave_test,4))
            if idx == 0:
                AUC_st_test += '{},'.format(round(AUC_ave_test,4))
            f1_st_test += '{},'.format(round(F1_ave_test,4))

            self.AUC_ave_dev_dict_list[top_K].append(AUC_ave_dev)
            self.recall_ave_dev_dict_list[top_K].append(recall_ave_dev)
            self.pre_ave_dev_dict_list[top_K].append(pre_ave_dev)
            self.F1_ave_dev_dict_list[top_K].append(F1_ave_dev)

            self.AUC_ave_test_dict_list[top_K].append(AUC_ave_test)
            self.recall_ave_test_dict_list[top_K].append(recall_ave_test)
            self.pre_ave_test_dict_list[top_K].append(pre_ave_test)
            self.F1_ave_test_dict_list[top_K].append(F1_ave_test)

            if AUC_ave_dev > self.AUC_ave_dev_dict_best[top_K]:
                self.AUC_ave_dev_dict_best[top_K] = AUC_ave_dev

            if recall_ave_dev > self.recall_ave_dev_dict_best[top_K]:
                self.recall_ave_dev_dict_best[top_K] = recall_ave_dev
                if self.save:
                    if top_K == 2:
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        filename = "{}/{}_{}_best.pth"\
                            .format(model_path, embedder, self.file_name)
                        print("Saving to {}".format(filename))
                        torch.save(model.state_dict(), filename)

            if pre_ave_dev > self.pre_ave_dev_dict_best[top_K]:
                self.pre_ave_dev_dict_best[top_K] = pre_ave_dev

            if F1_ave_dev > self.F1_ave_dev_dict_best[top_K]:
                self.F1_ave_dev_dict_best[top_K] = F1_ave_dev

            if AUC_ave_test > self.AUC_ave_test_dict_best[top_K]:
                self.AUC_ave_test_dict_best[top_K] = AUC_ave_test

            if recall_ave_test > self.recall_ave_test_dict_best[top_K]:
                self.recall_ave_test_dict_best[top_K] = recall_ave_test

            if pre_ave_test > self.pre_ave_test_dict_best[top_K]:
                self.pre_ave_test_dict_best[top_K] = pre_ave_test

            if F1_ave_test > self.F1_ave_test_dict_best[top_K]:
                self.F1_ave_test_dict_best[top_K] = F1_ave_test

            self.AUC_ave_dev_dict_list_best[top_K].append(self.AUC_ave_dev_dict_best[top_K])
            self.pre_ave_dev_dict_list_best[top_K].append(self.pre_ave_dev_dict_best[top_K])
            self.recall_ave_dev_dict_list_best[top_K].append(self.recall_ave_dev_dict_best[top_K])
            self.F1_ave_dev_dict_list_best[top_K].append(self.F1_ave_dev_dict_best[top_K])

            self.AUC_ave_test_dict_list_best[top_K].append(self.AUC_ave_test_dict_best[top_K])
            self.pre_ave_test_dict_list_best[top_K].append(self.pre_ave_test_dict_best[top_K])
            self.recall_ave_test_dict_list_best[top_K].append(self.recall_ave_test_dict_best[top_K])
            self.F1_ave_test_dict_list_best[top_K].append(self.F1_ave_test_dict_best[top_K])

            recall_st_dev_best += '{},'.format(round(self.recall_ave_dev_dict_list_best[top_K][-1], 4))
            precision_st_dev_best += '{},'.format(round(self.pre_ave_dev_dict_list_best[top_K][-1], 4))
            if idx == 0:
                AUC_st_dev_best += '{},'.format(round(self.AUC_ave_dev_dict_list_best[top_K][-1], 4))
            f1_st_dev_best += '{},'.format(round(self.F1_ave_dev_dict_list_best[top_K][-1], 4))

            recall_st_test_best += '{},'.format(round(self.recall_ave_test_dict_list_best[top_K][-1], 4))
            precision_st_test_best += '{},'.format(round(self.pre_ave_test_dict_list_best[top_K][-1], 4))
            if idx == 0:
                AUC_st_test_best += '{},'.format(round(self.AUC_ave_test_dict_list_best[top_K][-1], 4))
            f1_st_test_best += '{},'.format(round(self.F1_ave_test_dict_list_best[top_K][-1], 4))

        recall_st_dev = recall_st_dev[:-1]
        precision_st_dev = precision_st_dev[:-1]
        AUC_st_dev = AUC_st_dev[:-1]
        f1_st_dev = f1_st_dev[:-1]

        recall_st_test = recall_st_test[:-1]
        precision_st_test = precision_st_test[:-1]
        AUC_st_test = AUC_st_test[:-1]
        f1_st_test = f1_st_test[:-1]

        recall_st_dev_best = recall_st_dev_best[:-1]
        precision_st_dev_best = precision_st_dev_best[:-1]
        AUC_st_dev_best = AUC_st_dev_best[:-1]
        f1_st_dev_best = f1_st_dev_best[:-1]

        recall_st_test_best = recall_st_test_best[:-1]
        precision_st_test_best = precision_st_test_best[:-1]
        AUC_st_test_best = AUC_st_test_best[:-1]
        f1_st_test_best = f1_st_test_best[:-1]

        print("{} [K:{}] [DEV] AUC: {:2}, recall: {}, pre: {}, F1: {} || [TEST]  AUC: {}, recall: {}, pre: {}, F1: {}"
              .format(st, self.top_Ks, AUC_st_dev, recall_st_dev, precision_st_dev, f1_st_dev, AUC_st_test, recall_st_test, precision_st_test, f1_st_test))
        sys.stdout.flush()

        if epoch >= self.early_stop:
            if self.recall_ave_dev_dict_best[2] == self.recall_ave_dev_dict_list_best[2][epoch-self.early_stop]:
                print("[Final] {} [K:{}] [DEV] AUC: {:2}, recall: {}, pre: {}, F1: {} || [TEST]  AUC: {}, recall: {}, pre: {}, F1: {}"
                    .format(st, self.top_Ks, AUC_st_dev_best, recall_st_dev_best, precision_st_dev_best, f1_st_dev_best,
                            AUC_st_test_best, recall_st_test_best, precision_st_test_best, f1_st_test_best))
                sys.stdout.flush()
                return True

        return False

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s

    def compute_top_k(self, idx, score_list, pos_score_list, neg_score_list, pos_a_ids):
        for top_K in self.top_Ks:
            correct_num = 0
            # for AUC
            pair_num = 0
            correct_num_auc = 0
            score_threshold = score_list[- top_K - 1]

            for pos_score in pos_score_list:
                if pos_score > score_threshold:
                    correct_num += 1

                for jj in range(len(neg_score_list)):
                    pair_num += 1
                    if pos_score > neg_score_list[jj]:
                        correct_num_auc += 1

            if idx <= int(len(self.dataset.test_idx ) / 2):
                self.evaluate_p_num_dev_dict.setdefault(top_K, 0)
                self.evaluate_p_num_dev_dict[top_K] += 1

                self.recall_ave_dev_dict.setdefault(top_K, 0)
                self.recall_ave_dev_dict[top_K] += float(correct_num) / len(pos_a_ids)

                self.pre_ave_dev_dict.setdefault(top_K, 0)
                self.pre_ave_dev_dict[top_K] += float(correct_num) / top_K

                self.AUC_ave_dev_dict.setdefault(top_K, 0)
                self.AUC_ave_dev_dict[top_K] += float(correct_num_auc) / pair_num
            else:
                self.evaluate_p_num_test_dict.setdefault(top_K, 0)
                self.evaluate_p_num_test_dict[top_K] += 1

                self.recall_ave_test_dict.setdefault(top_K, 0)
                self.recall_ave_test_dict[top_K] += float(correct_num) / len(pos_a_ids)

                self.pre_ave_test_dict.setdefault(top_K, 0)
                self.pre_ave_test_dict[top_K] += float(correct_num) / top_K

                self.AUC_ave_test_dict.setdefault(top_K, 0)
                self.AUC_ave_test_dict[top_K] += float(correct_num_auc) / pair_num

    def compute_reciprocal(self, gt_lst, false_lst):
        incor_count=0
        for rank_f in false_lst:
            for rank_g in gt_lst:
                if rank_f < rank_g:
                    incor_count+=1

        return incor_count

    def evaluate_Camel(self, model, embedder, model_path, st, epoch, p_text_deep_f, a_latent_f):
        self.evaluate_p_num_dev_dict = {}
        self.recall_ave_dev_dict = {}
        self.pre_ave_dev_dict = {}
        self.AUC_ave_dev_dict = {}
        self.evaluate_p_num_test_dict = {}
        self.recall_ave_test_dict = {}
        self.pre_ave_test_dict = {}
        self.AUC_ave_test_dict = {}
        paper_list = self.dataset.test_idx
        for idx, target_paper in enumerate(paper_list):
            pos_a_ids = self.dataset.test_p_a_dir[target_paper]
            score_list = []

            # for AUC
            pos_score_list = []
            neg_score_list = []

            for author in pos_a_ids:
                score_temp = self.compute_score(p_text_deep_f[self.p_id_map[target_paper]], a_latent_f[author], metric=self.metric)
                score_list.append(score_temp)
                pos_score_list.append(score_temp)

            for a_id_temp in self.p_a_idx_neg_dict_test[target_paper]:
                score_temp = self.compute_score(p_text_deep_f[self.p_id_map[target_paper]], a_latent_f[a_id_temp], metric=self.metric)
                score_list.append(score_temp)
                neg_score_list.append(score_temp)

            score_list.sort()
            self.compute_top_k(idx, score_list, pos_score_list, neg_score_list, pos_a_ids)

        return self.print_result(model, embedder, model_path, epoch, st)


