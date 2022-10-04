from sklearn.metrics import accuracy_score, recall_score, f1_score
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