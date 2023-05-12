import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

seed = 888 # keep seed the same as in the training to avoid the undeterministic behaviour of the torch.
if cuda:  # Cuda will help if the machine contains NVIDIA GPU.
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)
print("Ready")

BATCH_SIZE = 10

data = fetch_california_housing()
print(data.feature_names)


class ManoDataSet(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        self.features_test = X
        self.target_test = Y
        assert self.features_test.shape[0] == self.target_test.shape[0]

    def __getitem__(self, index):
        feature = self.features_test[index]
        target = self.target_test[index]
        return feature, target

    def __len__(self):
        return self.target_test.shape[0]


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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


# Load model from pkl file
parameter = torch.load('./NetworkParameters.pkl', map_location=torch.device('cpu'))
model = CNN()
model.to(device)
model.load_state_dict(parameter)

loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


# Load test data from file
features_train, features_test, target_train, target_test = train_test_split(data.data, data.target, train_size=0.7,
                                                                            shuffle=True)

test_dateset = ManoDataSet(features_test, target_test)
test_loader = DataLoader(dataset=test_dateset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

history = []
plt.ion()

for step, (x, y) in enumerate(tqdm(test_loader)):
    features = x.type(torch.float32).to(device)
    y = y.type(torch.float32)
    labels = y.unsqueeze(1)
    prediction = model(features)
    loss = loss_func(prediction, labels)
    mse = float(loss)
    history.append(mse)

    plt.clf()
    plt.plot(range(len(history)), history)
    plt.pause(0.01)
    plt.savefig('testing')

print('over')
