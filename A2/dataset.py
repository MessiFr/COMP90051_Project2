import json
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
import copy
class Dataset:
    def __init__(self, data_file):
        with open(data_file, encoding='utf-8') as f:
            line = f.readline()
            self.data = json.loads(line)
            f.close()
        self.author_map = self.get_author_map()
        self.a_p, self.p_a_dir_dic = self.get_relation(self.data)
        self.p_a_indir_dic = self.paper_author_indirect_relation(self.p_a_dir_dic, self.a_p)
        self.author_num = len(self.a_p.keys())
        self.generate_train_test()
        self.context_data, self.title_data = self.get_context_data()
        self.word_n = max([max(x) for x in self.context_data])

        
    def get_author_map(self):
        authors = []
        for paper in self.data:
            authors.extend(paper["authors"])
        authors = sorted(list(set(authors)))
        author_map = {}
        for i in range(len(authors)):
            author_map[authors[i]] = i
        return author_map

    def get_relation(self, data):
        a_p = {}
        p_a = {}
        for i in range(len(data)):
            p_a[i] = []
            for author in data[i]["authors"]:
                mapped_author = self.author_map[author]
                p_a[i].append(mapped_author)
                if(mapped_author in a_p):
                    a_p[mapped_author].append(i)
                else:
                    a_p[mapped_author] = [i]
        return a_p, p_a
    
    def get_context_data(self):
        context, title = [],[]
        for paper in self.data:
            context.append(paper["abstract"])
            title.append(paper["title"])
        return context, title

    def paper_author_indirect_relation(self, p_a, a_p):
        indir_p_a = {}
        for paper in p_a:
            for author in p_a[paper]:
                for ind_paper in a_p[author]:
                    if(paper not in indir_p_a):
                        indir_p_a[paper] = copy.deepcopy(p_a[ind_paper])
                    else:
                        indir_p_a[paper].extend(p_a[ind_paper])
                indir_p_a[paper] = list(set(indir_p_a[paper]))
        return indir_p_a

    def generate_train_test(self):
        train_idx, test_idx = train_test_split([i for i in range(len(self.data))],random_state=1, test_size = 0.2)
        self.train_idx, self.test_idx = train_idx, test_idx
        self.train_p_a_dir = {}
        self.test_p_a_dir = {}
        self.train_p_a_indir = {}
        self.test_p_a_indir = {}
        # dir_auth = []
        # indir_auth = []
        for i in self.p_a_dir_dic:
            if(i in train_idx):
                self.train_p_a_dir[i] = copy.deepcopy(self.p_a_dir_dic[i])
                # dir_auth.extend(self.p_a_dir_dic[i])
                # dir_auth = list(set(dir_auth))
            else:
                self.test_p_a_dir[i] = copy.deepcopy(self.p_a_dir_dic[i])
        for i in self.p_a_indir_dic:
            if(i in train_idx):
                self.train_p_a_indir[i] = copy.deepcopy(self.p_a_indir_dic[i])
                # indir_auth.extend(self.p_a_indir_dic[i])
                # indir_auth = list(set(indir_auth))
            else:
                self.test_p_a_indir[i] = copy.deepcopy(self.p_a_indir_dic[i])
        # print("dir author", len(dir_auth))
        # print("indir author", len(indir_auth))
        # print("total author", self.author_num)

    def p_a_a_dir(self):
        p_a_a_dir = []
        for p_id, a_ids in self.train_p_a_dir.items():
            for a_pos in a_ids:
                a_neg = random.randint(0, self.author_num - 1)
                while (a_neg in a_ids):
                    a_neg = random.randint(0, self.author_num - 1)
                # triple = [p_id, int(a_pos[1:]), a_neg]
                triple = [p_id, a_pos, a_neg]
                p_a_a_dir.append(triple)
        return torch.LongTensor(p_a_a_dir)
    
    def p_a_a_indir(self):
        p_a_a_indir_list_batch = []
        dir_len = sum([len(self.train_p_a_dir[x]) for x in self.train_p_a_dir])
        indir_len = sum([len(self.train_p_a_indir[x]) for x in self.train_p_a_indir])
        p_threshold = float(dir_len) / indir_len + 3e-3
        for p_id, a_ids in self.train_p_a_indir.items():
            for a_pos in a_ids:
                if random.random() < p_threshold:
                    a_neg = random.randint(0, self.author_num - 1)
                    while (a_neg in a_ids):
                        a_neg = random.randint(0, self.author_num - 1)
                    triple = [p_id, a_pos, a_neg]
                    p_a_a_indir_list_batch.append(triple)
        return torch.LongTensor(p_a_a_indir_list_batch)
    
    def p_a_neg_dir_test(self):
        d ={}
        for i in self.test_p_a_dir:
            neg_num = 100 - len(self.test_p_a_dir[i])
            cur = []
            for _ in range(neg_num):
                neg_id = random.randint(0, self.author_num - 1)
                while (neg_id in self.test_p_a_dir[i]) or (neg_id not in self.a_p):
                    neg_id = random.randint(0, self.author_num - 1)
                cur.append(neg_id)
            d[i] = cur
        return d
    def get_content(self, papers):
        content = []
        for paper in papers:
            id = paper[0]
            content.append(self.context_data[id])
        return content
