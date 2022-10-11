import networkx as nx
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from preprocessing import for_train
# import tensorflow as tf
# import tensorflow_gnn as tfgnn
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

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
        
        # features = torch.tensor(features, dtype=torch.float32)
        features =  features.clone().detach()
        
        label_dict = {'features': features}

        for i in range(100):
            # label_dict[i] = torch.tensor(labels[i], dtype=torch.float32)
            label_dict[i] = labels[i].clone().detach()

        return label_dict
        
def BinaryDataLoader(X, y, shuffle=True, batch_size=1):
    dataset = BinaryDataset(X, y)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataset, dataloader

class SentenceRNN(nn.Module):
    def __init__(self, n_dim):
        super(SentenceRNN, self).__init__()

        self.hidden_size = 512
        self.num_layers = 2
        self.lstm = nn.LSTM(n_dim, 512, 2, batch_first=True)
        self.fc = nn.Linear(512, 100)
        # self.out = nn.Linear(128, 100)
    
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    
        outs, _ = self.lstm(x, (h0, c0))
        
        outs = self.fc(outs[:, -1, :])

        return outs


class NeuralNetworkCoauthor(nn.Module):
    def __init__(self, n_dim):
        super(NeuralNetworkCoauthor, self).__init__()

        ## 1024, 512, 256, 100
        ## nn.Dropout() // 0.1, 0.2 //

        self.fc1 = nn.Linear(n_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, 100)
        
        self.out = nn.Linear(512, 100)
    
    def forward(self, x):

        # F.leaky_relu()

        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        outs = torch.sigmoid(self.out(x)) 

        return outs

class NeuralNetworkYearVenue(nn.Module):
    def __init__(self, n_dim):
        super(NeuralNetworkYearVenue, self).__init__()

        ## 1024, 512, 256, 100
        ## nn.Dropout() // 0.1, 0.2 //

        self.fc1 = nn.Linear(n_dim, 256)    
        # self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(256, 100)

    def forward(self, x):

        # F.leaky_relu()

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        
        outs = torch.sigmoid(self.out(x)) 

        return outs

        

class NeuralNetworkYearVenueNoEmbedding(nn.Module):
    def __init__(self, n_dim):
        super(NeuralNetworkYearVenueNoEmbedding, self).__init__()

        ## 1024, 512, 256, 100
        ## nn.Dropout() // 0.1, 0.2 //

        self.fc1 = nn.Linear(n_dim, 64)  
        self.out = nn.Linear(64, 100)

    def forward(self, x):

        # F.leaky_relu()

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        
        outs = torch.sigmoid(self.out(x)) 

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

        elif loss_func == "MultiLabelMarginLoss":
            # outputs_ = outputs_.long()
            target = targets[i].long()
            sum_ += nn.MultiLabelMarginLoss()(outputs_, target)


    return sum_ / n_class
    
# training function
def train_steps(model, dataloader, optimizer, device, lstm=False, loss_func="BCE"):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for _, data in enumerate(dataloader):
    
        counter += 1
        
        # extract the features and labels
        features = data['features'].to(device)
        
        if lstm:
            features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))    
            
        targets = []
        for j in range(100):
            targets.append(data[j].to(device))
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        # print(features)
        outputs = model(features)
    
        loss = loss_fn(outputs, targets, loss_func)
        
    
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        
        # update optimizer parameters
        optimizer.step()

        
    train_loss = train_running_loss / counter
    return train_loss


class Model():
    
    def __init__(self) -> None:
        self.train_loss = []
        self.model = None
        self.device = None
        self.loss_fc = None
        self.lstm = False
        self.n_dim = 0
        

    def set_dim(self, n_dim):
        self.n_dim = n_dim

    def year_venue_model(self, embedding=True):
        if embedding:
            self.model = NeuralNetworkYearVenue(self.n_dim)
        else:
            self.model = NeuralNetworkYearVenueNoEmbedding(self.n_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fc = "BCE"
        return self.model
    
    def coauthor_model(self):
        self.model = NeuralNetworkCoauthor(self.n_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fc = "BCE"
        return self.model

    def abstract_title(self):
        self.model = SentenceRNN(self.n_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fc = "MSE"
        self.lstm = True
        return self.model


    def train(self, dataloader, optimizer, epochs):
        for _ in tqdm(range(epochs)):
    
            train_epoch_loss = train_steps(
                self.model, dataloader, optimizer, self.device, lstm=self.lstm, loss_func=self.loss_fc
            )
            self.train_loss.append(train_epoch_loss)
            
    def save_status(self, fileName):
        self.statusFile = fileName
        torch.save(self.model.state_dict(), f'status/{fileName}.pth')

    def plot_loss(self):
        # plot and save the train loss graph
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_loss, color='orange', label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.savefig('outputs/multi_head_binary_loss.png')
        plt.show()


def LogisticRegressionSplit(data, p=0.5):

    X, y_array = for_train('coauthor', p=p, type='numpy')

    y = []
    for i in y_array:
        if sum(i) > 0:
            y.append(1)
        else:
            y.append(0)

    clf = LogisticRegression(random_state=0).fit(X, y)
    
    print("Accuracy of split (prolific authors) & (no prolific authors)  : ", clf.score(X, y))

    y_pred = clf.predict(data)
    pa = []
    nopa = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            nopa.append(i)
        else:
            pa.append(i)

    return {'have_pauthor': pa, 'no_pauthor': nopa}

class GCN(nn.Module):
    def __init__(self, n_dim) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(n_dim, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, 128)
        self.classifier = nn.Linear(128, 100)

        # self.conv1 = GCNConv(n_dim, 4)
        # self.conv2 = GCNConv(4, 4)
        # self.conv3 = GCNConv(4, 2)
        # self.classifier = nn.Linear(2, 4)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = torch.sigmoid(self.classifier(h))

        return out, h


class GraphNeuralNetwork():

    def __init__(self) -> None:
        
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_loss = []
        self.X = None
        self.edges = None
        self.y = None
        self.mask = None

    def train_step(self):
        self.optimizer.zero_grad()
        out, h = self.model(self.X, self.edges)

        # loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss = self.criterion(out[self.mask], self.y[self.mask])

        loss.backward()
        self.optimizer.step()
        return loss, h

    def fit(self, X, edges, y, mask):
        self.model = GCN(X.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.X = X
        self.edges = edges
        self.y = y
        self.mask = mask

    def train(self, epochs=100):
        for _ in tqdm(range(epochs)):
            loss, _ = self.train_step()
            self.train_loss.append(loss.item())

    def plot_loss(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_loss, color='orange', label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        return self.model(X, self.edges)