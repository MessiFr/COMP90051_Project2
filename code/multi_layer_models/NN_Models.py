import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# `BinaryDataset()` class for multi-head binary classification model
class BinaryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]
        
        # we have 12 feature columns 
        features = torch.tensor(features, dtype=torch.float32)
        # there are 5 classes and each class can have a binary value ...
        # ... either 0 or 1

        label_dict = {'features': features, 'labels': labels}

        # for i in range(100):
        #     key = 'label' + str(i)
        #     label_dict[key] = torch.tensor(labels[i], dtype=torch.float32)

        return label_dict
        
def BinaryDataLoader(X, y, shuffle=True, batch_size=1):
    dataset = BinaryDataset(X, y)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataset, dataloader

class SentenceRNN(nn.Module):
    def __init__(self):
        super(SentenceRNN, self).__init__()

        self.hidden_size = 512
        self.num_layers = 2
        self.lstm = nn.LSTM(4999, 512, 2, batch_first=True)
        self.fc = nn.Linear(512, 100)
        # self.out = nn.Linear(128, 100)
    
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    
        outs, _ = self.lstm(x, (h0, c0))
        
        outs = self.fc(outs[:, -1, :])

        return outs


class NeuralNetworkCoauthor(nn.Module):
    def __init__(self):
        super(NeuralNetworkCoauthor, self).__init__()

        ## 1024, 512, 256, 100
        ## nn.Dropout() // 0.1, 0.2 //

        self.fc1 = nn.Linear(21146, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 100)
    
    def forward(self, x):

        # F.leaky_relu()

        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        outs = F.sigmoid(self.out(x)) 

        return outs

class NeuralNetworkYearVenue(nn.Module):
    def __init__(self):
        super(NeuralNetworkYearVenue, self).__init__()

        ## 1024, 512, 256, 100
        ## nn.Dropout() // 0.1, 0.2 //

        self.fc1 = nn.Linear(486, 256)
        self.out = nn.Linear(256, 100)
    
    def forward(self, x):

        # F.leaky_relu()

        x = F.relu(self.fc1(x))
        
        outs = F.sigmoid(self.out(x)) 

        return outs


# custom loss function for multi-head binary classification
# binary_loss_fn
def loss_fn(outputs, targets, loss_func="BCE"):

    n_class = 100
    sum_ = 0
    
    for i in range(n_class):
        
        outputs_ = outputs[:, i]
        
        if loss_func == "BCE":
            sum_ += nn.BCELoss()(outputs_, targets[i])
        
        ## 多标签分类 (mse) 
        elif loss_func == "MSE":
            sum_ += nn.MSELoss()(outputs_, targets[i])


    return sum_ / n_class
    
# training function
def train(model, dataloader, optimizer, loss_fn, train_dataset, device, lstm=False, loss_func="BCE"):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for _, data in tqdm(enumerate(dataloader), total=int(len(train_dataset)/dataloader.batch_size)):
    
        counter += 1
        
        # extract the features and labels
        features = data['features'].to(device)
        
        if lstm:
            features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))    
            
        # targets = []
        # for j in range(100):
        #     targets.append(data[f'label{j}'].to(device))
        targets = data['labels']
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(features)
        
        loss = loss_fn(outputs, targets, loss_func)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss