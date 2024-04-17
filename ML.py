import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

class ProteinDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.target[idx]


class ProteinModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_h = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc_h(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss.detach().cpu().numpy()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            

    test_loss /= len(test_loader.dataset)

    return test_loss


def train_and_test(model, device, train_loader, test_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train_model(model, device, train_loader, optimizer, criterion)
        
        test_loss = test_model(model, device, test_loader, criterion)
        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.4f}'.format(epoch, train_loss, test_loss))


def evaluate_model(model, device, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_true.extend(target.tolist())
            y_pred.extend(output.tolist())

    mse = mean_squared_error(y_true, y_pred), 
    r2 = r2_score(y_true, y_pred), 
    mae = mean_absolute_error(y_true, y_pred), 
    evs = explained_variance_score(y_true, y_pred)  
    

    return mse, r2, mae, evs


def load_data(data_path='/public/home/gongzhichen/code/data/tap.npz'):
    data = np.load(data_path, allow_pickle=True)
    train_X, valid_X, test_X, train_y, valid_y, test_y = data['train_X'], data['valid_X'], data['test_X'], data['train_Y'].astype(np.float32), data['valid_Y'].astype(np.float32), data['test_Y'].astype(np.float32)

    scaler = StandardScaler().fit(np.concatenate((train_X, valid_X, test_X), axis=0))

    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)
    
    return train_X, valid_X, test_X, train_y, valid_y, test_y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_dataloaders(train_X, valid_X, test_X, train_y, valid_y, test_y, batch_size=512):
    train_dataset = ProteinDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float())
    valid_dataset = ProteinDataset(torch.from_numpy(valid_X).float(), torch.from_numpy(valid_y).float())
    test_dataset = ProteinDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,valid_loader, test_loader


def create_model(input_size, hidden_size):
    model = ProteinModel(input_size, hidden_size)
    return model


def create_optimizer(model, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


def create_criterion():
    criterion = nn.MSELoss()
    return criterion


def run_ml(data_path, input_size, hidden_size, lr, batch_size, num_epochs):
    train_X, valid_X, test_X, train_y, valid_y, test_y = load_data(data_path)
    train_loader,valid_loader, test_loader = create_dataloaders(train_X, valid_X, test_X, train_y, valid_y, test_y, batch_size)
    model = create_model(input_size, hidden_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = create_optimizer(model, lr)
    criterion = create_criterion()
    train_and_test(model, device, train_loader, valid_loader, optimizer, criterion, num_epochs)
    mse, r2, mae, evs = evaluate_model(model, device, test_loader)
    print('mse: {}, r2: {}, mae: {}, evs: {}'.format(mse, r2, mae, evs))        

def run_svm(data_path, kernel='polynomial', C=0.1):
    train_X, valid_X, test_X, train_y, valid_y, test_y = load_data(data_path)
    clf = svm.SVR(kernel=kernel, C=C)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    evs = explained_variance_score(test_y, y_pred)
    print('mse: {}, r2: {}, mae: {}, evs: {}'.format(mse, r2, mae, evs))


def run_rf(data_path, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_samples=None):    
    train_X, valid_X, test_X, train_y, valid_y, test_y = load_data(data_path)
    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_samples=max_samples)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    evs = explained_variance_score(test_y, y_pred)
    print('mse: {}, r2: {}, mae: {}, evs: {}'.format(mse, r2, mae, evs))


def run_lr(data_path):
    train_X, valid_X, test_X, train_y, valid_y, test_y = load_data(data_path)
    clf = LinearRegression()
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    evs = explained_variance_score(test_y, y_pred)
    print('mse: {}, r2: {}, mae: {}, evs: {}'.format(mse, r2, mae, evs))


def run_dt(data_path, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    train_X, valid_X, test_X, train_y, valid_y, test_y = load_data(data_path)
    clf = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    evs = explained_variance_score(test_y, y_pred)
    print('mse: {}, r2: {}, mae: {}, evs: {}'.format(mse, r2, mae, evs))


if __name__ == '__main__':
    run_ml('/public/home/gongzhichen/code/data/tap.npz', 640, 64, 2e-5, 512, 5000)
    # run_svm('/public/home/gongzhichen/code/data/tap.npz', kernel='linear', C=1.0)
    # run_rf('/public/home/gongzhichen/code/data/tap.npz', n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_samples=None)
    # run_lr('/public/home/gongzhichen/code/data/tap.npz')
    # run_dt('/public/home/gongzhichen/code/data/tap.npz', max_depth=None, min_samples_split=2, min_samples_leaf=1)