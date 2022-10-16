import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BinaryDataset(Dataset):
    '''
    Set the dataset class
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]
        
        features =  features.clone().detach()
        
        label_dict = {'features': features, 'labels': labels}

        return label_dict
        
def BinaryDataLoader(X, y, shuffle=True, batch_size=1):
    '''
    Return the dataloader with input batch size
    Parameters:
        X: tensor([n_samples, n_features])
        y: tensor([n_samples, 100])
        shuffle: Boolean
        Batch_size: int (Default 1)
    '''
    dataset = BinaryDataset(X, y)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataset, dataloader

class SentenceRNN(nn.Module):
    '''
    RNN Model for training the abstract & title features
    '''
    def __init__(self, n_dim):
        super(SentenceRNN, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2

        self.lstm = nn.LSTM(n_dim, 128, 2, batch_first=True)
        self.out = nn.Linear(128, 100)
    
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    
        outs, _ = self.lstm(x, (h0, c0))
        
        outs = self.out(outs[:, -1, :])

        return outs


class NeuralNetworkCoauthor(nn.Module):
    def __init__(self, n_dim):
        super(NeuralNetworkCoauthor, self).__init__()

        self.fc1 = nn.Linear(n_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 100)
    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        outs = torch.sigmoid(self.out(x)) 

        return outs

class NeuralNetworkYearVenue(nn.Module):
    def __init__(self, n_dim):
        super(NeuralNetworkYearVenue, self).__init__()

        self.fc1 = nn.Linear(n_dim, 128)
        self.out = nn.Linear(128, 100)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        
        outs = torch.sigmoid(self.out(x)) 

        return outs


def loss_fn(outputs, targets, loss_func="BCE"):
    '''
    Return the loss using input loss function
    Parameters:
        outputs: tensor([n_samples, 100])
        targets: tensor([n_samples, 100])
        loss_func: string ('BCE' || 'MSE' || 'MultiLabelMarginLoss')
    '''

    loss = 0
    
    if loss_func == "BCE":
        loss = nn.BCELoss()(outputs, targets)
    
    ## (mse) 
    elif loss_func == "MSE":
        loss = nn.MSELoss()(outputs, targets)

    elif loss_func == "MultiLabelMarginLoss":
        # outputs_ = outputs_.long()
        target = targets.long()
        loss += nn.MultiLabelMarginLoss()(outputs, target)


    return loss
    

def train_steps(model, dataloader, optimizer, device, lstm=False, loss_func="BCE"):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for _, data in enumerate(dataloader):
    
        counter += 1
        
        # extract the features and labels
        features = data['features'].to(device)    
        targets = data['labels'].to(device)

        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(features)

        # get loss
        loss = loss_fn(outputs, targets, loss_func)
    
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        
        # update optimizer parameters
        optimizer.step()

        
    train_loss = train_running_loss / counter
    return train_loss


class Model():
    '''
    Class of Models, can used to define which neural network model should be use
    '''
    def __init__(self) -> None:
        self.train_loss = []
        self.model = None
        self.device = None
        self.loss_fc = None
        self.lstm = False
        self.n_dim = 0

    def set_dim(self, n_dim):
        self.n_dim = n_dim

    def year_venue_model(self):
        self.model = NeuralNetworkYearVenue(self.n_dim)
        
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

    def predict(self, X_test):
        logits = self.model(X_test)
        return logits

    
    def save_status(self, fileName):
        self.statusFile = fileName
        torch.save(self.model.state_dict(), f'status/{fileName}.pth')

    def plot_loss(self):
        plot_loss_graph(self.train_loss)


def plot_loss_graph(train_loss):
    '''
    Plot the loss function plots
    '''
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()