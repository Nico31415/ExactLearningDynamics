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

sigma=1



def reshape_matrix(input_matrix, new_shape):

    old_shape = input_matrix.shape
    

    if new_shape[0] > old_shape[0]:

        new_matrix = np.vstack((input_matrix, np.zeros((new_shape[0] - old_shape[0], old_shape[1]))))
    elif new_shape[0] < old_shape[0]:

        new_matrix = input_matrix[:new_shape[0], :]
    else:
        new_matrix = input_matrix
    
    if new_shape[1] > old_shape[1]:
        new_matrix = np.hstack((new_matrix, np.zeros((new_shape[0], new_shape[1] - old_shape[1]))))
    elif new_shape[1] < old_shape[1]:
        new_matrix = new_matrix[:, :new_shape[1]]
    
    return new_matrix



def balanced_weights(in_dim, hidden_dim, out_dim):
    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    w1 = sigma * np.random.randn(hidden_dim, in_dim)
    w2 = sigma * np.random.randn(out_dim, hidden_dim)

    U_, S_, V_ = np.linalg.svd(w2 @ w1)
    s = np.sqrt(np.diag(S_))

    lmda = np.trace(w2 @ w1) / hidden_dim

    factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

    s_2 = np.sqrt(np.diag(np.diag(factor)))

    s2_reshaped = reshape_matrix(s_2, (out_dim, hidden_dim))

    s_1 = np.diag(np.diag(s) / np.diag(s_2))

    s1_reshaped = reshape_matrix(s_1, (hidden_dim, in_dim))

    S_test = s2_reshaped @ s1_reshaped

    w1_out = r @ s1_reshaped @ V_.T 

    w2_out = U_ @ s2_reshaped @ r.T

    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    scale_by = lmda / q[0][0]
    w1_out = scale_by * w1_out
    w2_out = scale_by * w2_out
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    return w1_out, w2_out, S_test, q

# w1, w2, S_test, q = balanced_weights(in_dim = 4, hidden_dim = 2, out_dim=3)


# sns.heatmap(w1 @ w1.T - w2.T @ w2 )

in_dim = 5
hidden_dim = 10
out_dim = 2




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
    print('first w1: ', model.first_layer.weight.data.clone())
    w_twos = [model.second_layer.weight.data.clone()]
    losses = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

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

    # print('X standardised: ', X_standardised)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    X_whitened = torch.tensor(X_pca / np.sqrt(pca.explained_variance_), dtype=torch.float32)

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = torch.randn(batch_size, in_dim)
    print(X)
    Y = torch.randn(batch_size, out_dim)
    print(Y)
    X_whitened = whiten(X)

    return X_whitened.clone(), Y.clone()





tau = 0.1
batch_size = 10
epochs = 400

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
init_w1 = torch.tensor(init_w1)
init_w2 = torch.tensor(init_w2)

model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1, init_w2=init_w2)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=tau)

X_train, Y_train = get_random_regression_task(batch_size, in_dim, out_dim)

print('init_w1: ', init_w1)
w1s, w2s, losses = train(model, X_train, Y_train, criterion, optimizer, epochs)

print('w1s[0]: ', w1s[0])

analytical = QQT(init_w1, init_w2, X_train.T, Y_train.T, True)
analytical = np.asarray([analytical.forward(tau) for _ in range(epochs)])

w2w1s = [w2 @ w1 for (w2, w1) in zip(w2s, w1s)]
diffs = [np.linalg.norm(w2w1 - anal) for (w2w1, anal) in zip(w2w1s, analytical)]


top_left_empirical = [w2w1[0][0] for w2w1 in w2w1s]
top_left_analytical = [anal[0][0] for anal in analytical]

plt.plot(top_left_analytical, label='analytical')
plt.plot(top_left_empirical, label='empirical')
plt.legend()
plt.show()

print(w2w1s[0])

anal_loss = np.linalg.norm(X_train @ torch.tensor(analytical[-1]).float().T - Y_train)
emp_loss = np.linalg.norm(X_train @ torch.tensor(w2w1s[-1]).T - Y_train)


output_analytical = [(X_train) @ torch.tensor(anal).float().T for anal in analytical]
output_empirical = [(X_train) @ torch.tensor(w2w1).float().T for w2w1 in w2w1s]


# print('final loss analytical: ', np.linalg.norm(anal_loss))
# print('final loss empirical: ', np.linalg.norm(emp_loss))

##PLOTTING TRAJECTORIES 
# plot_items_n = 3


# empirical_outputs = (w2w1s @ X_train[:plot_items_n].T)
# analytical_outputs = ...

# blind_colours = BlindColours().get_colours()

# fig, ax = plt.subplots(1, 1, figsize=(12,3.))
# for color, output in zip(blind_colours, empirical_outputs.T):
#     for val in output:
#         ax.plot(val, c=color, lw=2.5)
# ax.plot((analytical @ X[:plot_items_n].T).reshape(epochs, -1), lw=3, c='k',alpha=0.7,linestyle=(0,(1,2)))

# ax.set_xscale('log')
# ax.set_xlabel('Training Steps')


# plt.show()

# outputs 



init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
init_w1 = torch.tensor(init_w1)
init_w2 = torch.tensor(init_w2)

model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1, init_w2=init_w2)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=tau)

num_tasks = 3

tasks = [get_random_regression_task(batch_size, in_dim, out_dim) for _ in range(num_tasks)]

w1s_list = []
w2s_list = []

for (X, Y) in tasks:
    w1s, w2s, _ = train(model, X, Y, criterion, optimizer, epochs)
    w1s_list.extend(w1s)
    w2s_list.extend(w2s)


w2w1s_list = [w2 @ w1 for (w2, w1) in zip(w2s_list, w1s_list)]

losses_list = []

for i, (X, Y) in enumerate(tasks):
    output_empirical = [X @ w2w1.T for w2w1 in w2w1s_list]
    losses = [0.5 * out_dim / batch_size * np.linalg.norm(out - Y) for out in output_empirical]
    losses_list.append(losses)

plt.plot(losses_list[1])
plt.title('Losses plotted')
plt.show()
print(w1s)


 

# plt.plot(diffs)
# plt.show()
