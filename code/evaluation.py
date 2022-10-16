from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from preprocessing import *
from NN_Models import *

def logits_to_matrix(logits, threshold):
    predict = np.zeros(logits.shape)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if logits[i, j] >= threshold:
                predict[i, j] += 1
    return predict

def get_f1(logits, y_test, threshold):
    '''
    logits: tensor([n_sample, 100])
    y_test: tensor([n_sample, 100]) 
    threshold: int
    '''

    predict = logits_to_matrix(logits, threshold)

    return f1_score(np.array(y_test), predict, average='samples', zero_division=1)

def evaluation(logits, y_test, thresholds):
    '''
    logits: tensor([n_sample, 100])
    y_test: tensor([n_sample, 100])
    thresholds: int[]
    '''
    f1_scores = []

    for i in tqdm(range(len(thresholds))):
        threshold = thresholds[i]

        f1 = get_f1(logits, y_test, threshold)

        f1_scores.append(f1)
    
    return f1_scores



def kaggle_predict(model, X_kaggle, fileName):
    '''
    model: model()
    X_kaggle: tensor([n_sample, n_feature])
    fileName: string | save the logits output in 'outputs/{fileName}.pt'
    '''

    logits = model.predict(X_kaggle)    

    torch.save(logits, f'outputs/{fileName}.pt')

    return 

def predict(author, COAUTHOR_WEIGHT, year_venue, YEAR_VENUE_WEIGHT, abstracts_title, SENTENCE_WEIGHT, THRESHOLD):
    '''
    author: predict logits of author, [n_samples, 100]
    COAUTHOR_WEIGHT: float
    year_venue: predict logits of year & venue, [n_samples, 100]
    YEAR_VENUE_WEIGHT: float
    abstracts_title: predict logits of abstracts_title, [n_samples, 100]
    SENTENCE_WEIGHT: float
    THRESHOLD: int
    '''
    
    weighted = author * COAUTHOR_WEIGHT + abstracts_title * SENTENCE_WEIGHT + year_venue * YEAR_VENUE_WEIGHT

    predict = logits_to_matrix(weighted, THRESHOLD)

    return predict

def to_list(matrix, threshold=0.5):
    '''
    Transfer the matrix of label (y) to list of stirng (test prolific authors)
    '''
    res_list = []
    for i in range(len(matrix)):
        pred = ""

        for j in range(100):
            val = matrix[i, j].item()
            
            if val >= threshold:
                pred += str(j) + " "

        if pred:
            res_list.append(pred[:-1])
        else:
            res_list.append("-1")

    return res_list

def print_scores(y_test, y_pred):
    print('='*25 + 'Evaluation results' + '='*25)
    print('The accuracy score of prediction is : {}'.format(accuracy_score(y_test, y_pred)))
    print('The recall   score of prediction is : {}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('The f1       score of prediction is : {}'.format(f1_score(y_test, y_pred, average='weighted')))

    

