import json
from tqdm import tqdm
import torch

# read train data and test data
f_train = open("../../data/train.json", 'r')
train_data = json.load(f_train)

f_test = open("../../data/test.json", 'r')
test_data = json.load(f_test)


def find_discard_authors(data):
    n_samples = len(data)
    empty_idx = []

    for i in tqdm(range(n_samples), desc="delet some useless data"):
        authors = data[i]['authors']
        p_author = 0

        for au in authors:
            if au < 100:
                p_author += 1
        
        if p_author == 0:
            empty_idx.append(i)
    
    print("Number of instance with label : ", n_samples-len(empty_idx))
    
    remain = int((0.20250*(n_samples-len(empty_idx))/(1-0.20250)))
    
    print("Number of instance without label(remain) : ", remain)
    
    discard_idx = empty_idx[remain:]

    return discard_idx


def get_word_matrix(data, discard_idx=[], train=True):
    n_samples = len(data)
    n_features = 5000 -1 

    # get abstract & title feature
    # 序列信息 ！！！！
    # 语言模型
    # lstm （*）

    #   XX bert pre-trained / 去掉embedding层 **

    if train:
        wmatrix = torch.zeros([n_samples-len(discard_idx), n_features])
    else:
        wmatrix = torch.zeros([n_samples, n_features])

    INDEX = 0
    for i in tqdm(range(n_samples), desc="title & abstract"):

        if i in discard_idx and train:
            continue

        instance = data[i]

        for title in instance['title']:
            wmatrix[INDEX, title-1] += 1
        for abstract in instance['abstract']:
            wmatrix[INDEX, abstract-1] += 1
        INDEX += 1
        
    return wmatrix


def get_author_matrix(data, discard_idx=[], train=True):
    n_samples = len(data)

    # get prolific authors 
    # co author -> feature / 单独编码
    # 多任务 - predict all authors 


    if train:
        y = torch.zeros([n_samples-len(discard_idx), 100])
        key = 'authors'
    else:
        y = None

    # get co-author matrix
    if train:
        amatrix = torch.zeros([n_samples-len(discard_idx), 21245 - 100 + 1])
    else:
        amatrix = torch.zeros([n_samples, 21245 - 100 + 1])
        key = 'coauthors'

    INDEX = 0
    for i in tqdm(range(n_samples), desc="authors"):
        if i in discard_idx and train:
            continue

        authors = data[i][key]
        
        for au in authors:
            if au < 100:
                y[INDEX, au] += 1

            else:
                amatrix[INDEX, au - 100] += 1

        INDEX += 1
        
    return amatrix, y

def get_year_venue_matrix(data, discard_idx=[], train=True):
    n_samples = len(data)

    # get venue feature
    # embedding

    if train:
        vmatrix = torch.zeros([n_samples-len(discard_idx), 466])
    else:
        vmatrix = torch.zeros([n_samples, 466])

    INDEX = 0
    for i in tqdm(range(n_samples), desc="venue"):
        if i in discard_idx and train:
            continue

        venue = data[i]['venue']
        
        if venue:
            vmatrix[INDEX, venue] = 1
        else:
            vmatrix[INDEX, 465] = 1
        INDEX += 1

    # get year feature !!!!!
    # 1-d 编码器 年份 输出特征 -》 256维
    # distribution of 
    # nn.Embedding() // input: batch-size * 1 // output: batch-size * embedding-size

    if train:
        ymatrix = torch.zeros([n_samples-len(discard_idx), 20])
    else:
        ymatrix = torch.zeros([n_samples, 20])
    

    INDEX = 0
    for i in tqdm(range(n_samples), desc="year"):
        if i in discard_idx and train:
            continue

        year = data[INDEX]['year']
        
        if year:
            ymatrix[INDEX, year] = 1
        else:
            ymatrix[INDEX, year] = 0
        INDEX += 1

    return torch.cat((vmatrix, ymatrix), 1)

def for_train(feature):

    di = find_discard_authors(train_data)

    if feature == 'coauthor':
        return get_author_matrix(train_data, discard_idx=di)

    elif feature == 'year_venue':
        X_all = get_year_venue_matrix(train_data, discard_idx=di)

    elif feature == 'word':
        X_all = get_word_matrix(train_data, discard_idx=di)

    _, y_all = get_author_matrix(train_data, discard_idx=di)

    return X_all, y_all

def for_kaggle(feature):

    if feature == 'coauthor':
        
        X_kaggle, _ = get_author_matrix(test_data, train=False)

    elif feature == 'year_venue':
        
        X_kaggle = get_year_venue_matrix(test_data, train=False)

    elif feature == 'word':
        
        X_kaggle = get_word_matrix(test_data, train=False)

    return X_kaggle


def transform_to_label(data, threshold):
    
    tmp = ""
    
    for i in range(100):
        if data[i] >= threshold:
            tmp += str(i) + " "
    if tmp:
        return tmp[: -1]
    else:
        return "-1"

def transform_labels(data, threshold):
    
    labels = []

    for i in range(data.shape[0]):
        labels.append(transform_to_label(data[i], threshold))
    
    return labels