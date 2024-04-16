import torch
import torch.nn as nn 
import torch.optim as optim

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import seaborn as sns
from tools import BlindColours, zero_balanced_weights
from dynamics import QQT

#THE NETWORK'S WEIGHTS ARE OPTIMISED WITH FULL BATCH GRADIENT DESCEINT WITH LEARNING RATE TAU

class LinearNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1=None, init_w2=None):
        super(LinearNetwork, self).__init__()
        self.first_layer = nn.Linear(in_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim , out_dim)

        if init_w1 is not None:
            self.first_layer.weight.data = init_w1.float()
        if init_w2 is not None:
            self.second_layer.weight.data = init_w2.float()

    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x 
    

def train(model, X_train, y_train, criterion, optimizer, epochs):
    
    w_ones = [model.first_layer.weight.data.clone()]
    w_twos = [model.second_layer.weight.data.clone()]
    losses = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_ones.append(model.first_layer.weight.data.clone().numpy())
        w_twos.append(model.second_layer.weight.data.clone().numpy())
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    
    w_ones = np.array(w_ones)
    w_twos = np.array(w_twos)
    losses = np.array(losses)

    return w_ones, w_twos, losses

def whiten(X):

    scaler = StandardScaler()
    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    X_whitened = torch.tensor(X_pca / np.sqrt(pca.explained_variance_), dtype=torch.float32)

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = torch.randn(batch_size, in_dim)
    Y = torch.randn(batch_size, out_dim)
    X_whitened = whiten(X)

    return X_whitened, Y


in_dim = 4
hidden_dim = 4
out_dim = 4

tau = 0.01
batch_size = 10
epochs = 400

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
init_w1 = torch.tensor(init_w1)
init_w2 = torch.tensor(init_w2)

model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1, init_w2=init_w2)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=tau)

X_train, Y_train = get_random_regression_task(batch_size, in_dim, out_dim)

w1s, w2s, losses = train(model, X_train, Y_train, criterion, optimizer, epochs)

analytical = QQT(init_w1, init_w2, X_train.T, Y_train.T, True)
analytical = np.asarray([analytical.forward(tau) for _ in range(epochs)])

w2w1s = [w2 @ w1 for (w2, w1) in zip(w2s, w1s)]
diffs = [np.linalg.norm(w2w1 - anal) for (w2w1, anal) in zip(w2w1s, analytical)]


top_left_empirical = [w2w1[0][0] for w2w1 in w2w1s]
top_left_analytical = [anal[0][0] for anal in analytical]



#TODO plot trajectories of N items
plt.plot(top_left_analytical, label='analytical')
plt.plot(top_left_empirical, label='empirical')
plt.title('Trajectories of top left item of both representations')
plt.legend()
plt.show()

anal_loss = np.linalg.norm(X_train @ torch.tensor(analytical[-1]).float().T - Y_train)
emp_loss = np.linalg.norm(X_train @ torch.tensor(w2w1s[-1]).T - Y_train)



output_analytical = [(X_train) @ torch.tensor(anal).float().T for anal in analytical]
output_empirical = [(X_train) @ torch.tensor(w2w1).float().T for w2w1 in w2w1s]

losses_empirical = [0.5 * out_dim / batch_size * np.linalg.norm(out - Y_train) for out in output_empirical]
losses_analytical = [0.5 * out_dim / batch_size * np.linalg.norm(out - Y_train) for out in output_analytical]


plt.plot(losses_analytical, label='analytical')
plt.plot(losses_empirical, label='empirical')
plt.title('Loss trajectories for Analytical and Empirical Gradient Flow')
plt.legend()
plt.show()

