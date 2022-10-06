from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from preprocessing import *
from NN_Models import *

def get_f1(model, X_test, y_test, threshold):
    
    _, test_dataloader = BinaryDataLoader(X_test, y_test, shuffle=False, batch_size=1)

    predict_list = []

    for _, test_sample in enumerate(test_dataloader):
        
        features = test_sample['features']
        
        
        features = torch.reshape(features, (features.shape[0], 1, features.shape[1])).to(device)
        
        outputs = model(features)
        
        outputs = outputs.squeeze()
                
        # get all the labels
        predict_list.append(transform_to_label(outputs, threshold=threshold))


    target_list = transform_labels(y_test, threshold=1)

    return predict_list, f1_score(target_list, predict_list, average='weighted')

def evaluation(model, X_test, y_test, thresholds):
    f1_scores = []

    for i in tqdm(range(len(thresholds))):
        threshold = thresholds[i]

        _, f1 = get_f1(model, X_test, y_test, threshold)

        f1_scores.append(f1)
    
    return f1_scores


def kaggle_predict(model, X_kaggle, fileName, lstm=False):
    predict_dict = {}

    key = 0
    for test_sample in tqdm(X_kaggle):
        
        features = test_sample
        if lstm:
            features = torch.reshape(features, (1, 1, 4999))
        
        outputs = model(features)
        outputs = outputs.squeeze()
        tmp = {}
        
        for i in range(100):
            tmp[i] = outputs[i].item()
            
        predict_dict[key] = tmp
        
        key += 1

    with open(f'outputs/{fileName}.json', 'w') as fp:
        json.dump(predict_dict, fp)

def predict(author, COAUTHOR_WEIGHT, year_venue, YEAR_VENUE_WEIGHT, abstracts_title, SENTENCE_WEIGHT, THRESHOLD):
    def get_weighted_value(i, j):
        val1 = year_venue[i][j] * YEAR_VENUE_WEIGHT
        val2 = author[i][j] * COAUTHOR_WEIGHT
        val3 = abstracts_title[i][j] * SENTENCE_WEIGHT
        
        return val1 + val2 + val3

    kaggle_predict = []

    for i in range(len(author)):
        # if i in has_author_dict['no_pauthor']:
        #     kaggle_predict.append('-1')
        #     continue
        
        pred = ""
        
        for j in range(100):
            
            val = get_weighted_value(str(i), str(j))
            
            if val >= THRESHOLD:
                pred += str(j) + " "

        if pred:
            kaggle_predict.append(pred[:-1])
        else:
            kaggle_predict.append("-1")

    return kaggle_predict

def to_list(matrix):
    res_list = []
    for i in range(len(matrix)):
        pred = ""

        for j in range(100):
            val = matrix[i, j].item()
            
            if val == 1:
                pred += str(j) + " "

        if pred:
            res_list.append(pred[:-1])
        else:
            res_list.append("-1")

    return res_list



class LogisticRegressionPredictModel():
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=0)

    def filter_data(self, dict1, dict2, dict3, y):
        n_sample = len(dict1.keys())

        X_new = np.ndarray([100 * n_sample, 3])
        
        y_new = np.ndarray([100 * n_sample, 1])

        for i in tqdm(range(n_sample)):
            # index = i // 10
            for j in range(100):
                X_new[i * 100 + j, 0] = dict1[str(i)][str(j)]
                X_new[i * 100 + j, 1] = dict2[str(i)][str(j)]
                X_new[i * 100 + j, 2] = dict3[str(i)][str(j)]
                y_new[i * 100 + j, 0] = y[i, j].item()

        count1 = 0
        index_list = []
        
        for i in range(100 * n_sample):
            if y_new[i, 0] > 0:
                index_list.append(i)
                count1 += 1

        count2 = 0
        for i in range(100 * n_sample):
            if i not in index_list:
                index_list.append(i)
                count2 += 1
                if count2 >= count1:
                    break

        X_new, y_new = X_new[index_list, :], y_new[index_list, :].squeeze()
        return X_new, y_new


    def train(self, dict1, dict2, dict3, y):
        X_new, y_new = self.filter_data(dict1, dict2, dict3, y)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)

        self.model.fit(X_train, y_train)
        print("Score : ", self.model.score(X_test, y_test))
        
        self.model.fit(X_new, y_new)

        return self.model
    
    def evaluation(self, dict1, dict2, dict3):
        
        pred_list = []
        n_sample = len(dict1.keys())
        for i in tqdm(range(n_sample)):
            author = ""
            for j in range(100):
                x1 = dict1[str(i)][str(j)]
                x2 = dict2[str(i)][str(j)]
                x3 = dict3[str(i)][str(j)]
                
                x = np.array([[x1, x2, x3]])
                au = self.model.predict(x)
                
                if au[0]:
                    author += str(j) + " "
                    
            if author:
                pred_list.append(author[:-1])
            else:
                pred_list.append("-1")

        return pred_list

    def score(self, dict1, dict2, dict3, y):
        y_test = to_list(y)
        print_scores(y_test, self.evaluation(dict1, dict2, dict3))


def print_scores(y_test, y_pred):
    print('='*25 + 'Evaluation results' + '='*25)
    print('The accuracy score of prediction is : {}'.format(accuracy_score(y_test, y_pred)))
    print('The recall   score of prediction is : {}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('The f1       score of prediction is : {}'.format(f1_score(y_test, y_pred, average='weighted')))

    