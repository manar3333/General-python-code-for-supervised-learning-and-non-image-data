import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
seed = 888
LearningRate = 0.0001
Batch = 10

# Obtaining the dataset
data = fetch_california_housing()
print(data.feature_names)

# Splitting the dataset
features_train, features_test, target_train, target_test = train_test_split(data.data, data.target, train_size=0.7,
                                                                            shuffle=True)


class ManoDataset(Dataset):
    # required for DataLoader
    def __init__(self, X, Y):
        self.features_train = X
        self.target_train = Y
        assert self.features_train.shape[0] == self.target_train.shape[0]

    def __getitem__(self, index):
        feature = self.features_train[index]
        target = self.target_train[index]
        return feature, target

    def __len__(self):
        return self.target_train.shape[0]


class NN(torch.nn.Module):
    # Our Pyramid structure neural network
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


training_dataset = ManoDataset(features_train, target_train)
trainLoader = DataLoader(dataset=training_dataset, batch_size=Batch, shuffle=True, drop_last=True)


net = NN()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

loss_recoder = []
plt.ion()

for i in range(100):
    for step, (x, y) in enumerate(tqdm(trainLoader)):

        features = x.type(torch.float32).to(device)
        y = y.type(torch.float32)
        labels = (y.unsqueeze(1)).to(device)

        prediction = net(features)

        loss = loss_func(prediction, labels)
        print("Epoch:", i, "prediction-->", prediction.cpu().detach().squeeze().numpy()[0], "labels--->", labels[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(net.state_dict(), 'NetworkParameters.pkl')

        loss_recoder.append(loss.cpu().detach().squeeze().numpy())

        plt.clf()
        plt.plot(range(len(loss_recoder)), loss_recoder)
        plt.pause(0.01)
        plt.savefig('loss')
print('over')
