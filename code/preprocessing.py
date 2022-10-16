from gensim.models.doc2vec import Doc2Vec
import json
from tqdm import tqdm
import torch
import random

# read train data and test data
with open("../../data/train.json", 'r') as f:
    train_data = json.load(f)
with open("../../data/test.json", 'r') as f:
    test_data = json.load(f)


def for_train(feature, di):
    '''
    Return the embedded feature of train data by input
    Parameters:
        feature: string 
            'coauthor' for coauthor feature
            'year_venue' for year and venue feature
            'word' for abstract and title feature, and return y as the labels
    '''

    if feature == 'coauthor':
        return get_author_matrix(train_data, discard_idx=di)

    elif feature == 'year_venue':
        X_all = get_year_venue_matrix(train_data, discard_idx=di)

    elif feature == 'word':
        X_all = get_word_matrix(train_data, discard_idx=di)

    _, y_all = get_author_matrix(train_data, discard_idx=di)

    return X_all, y_all

def for_kaggle(feature):
    '''
    Return the embedded feature of test data by input
    Parameters:
        feature: string 
            'coauthor' for coauthor feature
            'year_venue' for year and venue feature
            'word' for abstract and title feature, and return y as the labels
    '''

    if feature == 'coauthor':
        
        X_kaggle, _ = get_author_matrix(test_data, train=False)

    elif feature == 'year_venue':
        
        X_kaggle = get_year_venue_matrix(test_data, train=False)
        
    elif feature == 'word':
        
        X_kaggle = get_word_matrix(test_data, train=False)

    return X_kaggle



def get_word_matrix(data, discard_idx=[], train=True):
    '''
    Return the embedded tensor feature of abstract & title 
    Parameters:
        data: json file
        discard_idx: int[] | discard index
        train: Boolean | True for train data or False for test data
    '''

    n_samples = len(data)
    
    d2v_abstract = Doc2Vec.load('data/doc2vec_abstract.model')
    d2v_title = Doc2Vec.load('data/doc2vec_title.model')
    
    wmatrix = []
            
    for i in tqdm(range(n_samples), desc="title & abstract"):

        if i in discard_idx and train:
            continue

        title = [str(j) for j in data[i]['title']]
        title = list(d2v_title.infer_vector(title))

        abstract = [str(j) for j in data[i]['abstract']]
        abstract = list(d2v_abstract.infer_vector(abstract))
        
        sentence = title + abstract
        
        wmatrix.append(sentence)
        
    return torch.tensor(wmatrix)


def get_author_matrix(data, discard_idx=[], train=True):
    '''
    Return the embedded tensor feature of coauthor and the one-hot embedding of author as the label
    Coauthors are embedded by one-hot
    Parameters:
        data: json file
        discard_idx: int[] | discard index
        train: Boolean | True for train data or False for test data
    '''

    n_samples = len(data)

    if train:
        y = torch.zeros([n_samples-len(discard_idx), 100])
        key = 'authors'
    else:
        y = None
        key = 'coauthors'

    # get co-author matrix
    if train:
        amatrix = torch.zeros([n_samples-len(discard_idx), 21245 - 100 + 1])

    else:
        amatrix = torch.zeros([n_samples, 21245 - 100 + 1])
        

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
    '''
    Return the embedded tensor feature of year & venue
    Venue is embedded by metapath2vec
    Year is embedded by one-hot
    Parameters:
        data: json file
        discard_idx: int[] | discard index
        train: Boolean | True for train data or False for test data
    '''

    n_samples = len(data)

    metapath2vec_paper = torch.load('data/metapath2vec_paper.pt')
    metapath2vec_venue = torch.load('data/metapath2vec_venue.pt')

    vmatrix = []
        
    for i in tqdm(range(n_samples), desc="venue"):
        if i in discard_idx and train:
            continue

        venue = data[i]['venue']
        
        if venue:
            if train:
                vmatrix.append(metapath2vec_paper[i] + metapath2vec_venue[venue]) 
            else:
                vmatrix.append(metapath2vec_paper[i-800] + metapath2vec_venue[venue]) 
                
        else:
            if train:
                vmatrix.append(metapath2vec_paper[i])
            else:
                vmatrix.append(metapath2vec_paper[i-800])
    
    vmatrix = torch.stack(vmatrix)
    year = torch.zeros([n_samples-len(discard_idx), 20])
    INDEX = 0
    for i in tqdm(range(n_samples), desc="year"):
        if i in discard_idx and train:
            continue
        year[INDEX, data[i]['year']] += 1
        INDEX += 1

    return torch.concat((vmatrix, year), axis=1)


def get_discard(p):
    '''
    Return the discard index from train.json 
    Parameters:
        data: json
        p: int 
            - probability of instance without prolific authors
            - p = number of instance without prolific authors for train / total number of training instance
    '''
    discard = find_discard_authors(train_data, p)
    dis_dict = {'discard_index': discard}
    with open('data/discard_index.json', 'w') as f:
        json.dump(dis_dict, f)

    return discard

def find_discard_authors(data, p):
    '''
    Return the discard index
    '''
    
    n_samples = len(data)
    empty_idx = []
    if p == -1:
        return []

    for i in tqdm(range(n_samples), desc="delet some useless data"):
        authors = data[i]['authors']
        p_author = 0

        for au in authors:
            if au < 100:
                p_author += 1
        
        if p_author == 0:
            empty_idx.append(i)
    
    print("Number of instance with label : ", n_samples-len(empty_idx))
    
    remain = int((p*(n_samples-len(empty_idx))/(1-p)))
    
    print("Number of instance without label(remain) : ", remain)

    return random.sample(empty_idx, len(empty_idx) - remain)


def transform_to_label(logits, threshold):
    '''
    Transform logit to label
    '''
    
    tmp = ""
    
    for i in range(100):
        if logits[i] >= threshold:
            tmp += str(i) + " "
    if tmp:
        return tmp[: -1]
    else:
        return "-1"

def transform_labels(logits, threshold):
    '''
    Transform logits matrix to list of labels
    '''
    
    labels = []

    for i in range(logits.shape[0]):
        labels.append(transform_to_label(logits[i], threshold))
    
    return labels
