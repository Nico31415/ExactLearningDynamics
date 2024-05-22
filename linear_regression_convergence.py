import gooseberry as gs
from dynamics import QQT
from tools import BlindColours, zero_balanced_weights

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn 
import torch.optim as optim

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import seaborn as sns
from tools import BlindColours, zero_balanced_weights

from empiricalTest import QQT_new
from empiricalTest import LinearNetwork
from empiricalTest import get_random_regression_task


epsilon = 1e-2

in_dim = 4
hidden_dim = 5
out_dim = 3

num_trials = 1000


batch_size = 10
learning_rate = 0.1
training_steps = 400

def initialize_random_weights(in_dim, hidden_dim, out_dim):
    init_w1 = np.random.rand(hidden_dim, in_dim)
    init_w2 = np.random.rand(out_dim, hidden_dim)
    return init_w1, init_w2

init_weights = [initialize_random_weights(in_dim, hidden_dim, out_dim) for _ in range(num_trials)]
# init_weights = [zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35) for _ in range(num_trials)]

##TODO just make random weights
##TODO try with batch_size 1 and L2 norm

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)



w_star = Y @ X.T @ np.linalg.inv(X @ X.T)
threshold_loss = 1/(2*batch_size) * np.linalg.norm(Y - w_star @  X, ord='fro')**2

distance_timestep_pairs = []
loss_diff_pairs = []


def find_first_index(losses, threshold_loss, epsilon):
    for i in range(len(losses)):
        if np.abs(losses[i] - threshold_loss) < epsilon:
            return i 
    return -1

for n in range(num_trials):
    init_w1, init_w2 = init_weights[n]
    dist = np.linalg.norm(init_w2 @ init_w1 - w_star, ord='fro')**2
    model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1.copy(), init_w2=init_w2.copy())
    w1s, w2s, _ = model.train(X, Y, training_steps, learning_rate) 
    losses = [1/(2 * batch_size) * np.linalg.norm(w2@w1@X - Y, ord='fro')**2 for (w2, w1) in zip(w2s, w1s)]
    loss_diff = losses[0] - losses[-1]
    index = find_first_index(losses, threshold_loss, epsilon)
    loss_diff_pairs.append((loss_diff, index))
    distance_timestep_pairs.append((dist, index))


distances = [pair[0] for pair in distance_timestep_pairs]
loss_diff = [pair[0] for pair in loss_diff_pairs]

timesteps = [pair[1] for pair in distance_timestep_pairs]


gradient = 1/(2*learning_rate*epsilon)
x_line = np.linspace(0, np.max(distances), 100)
y_line = gradient * x_line

# mu = 1/batch_size
# upper_bound2 = 1/np.log(1-mu * learning_rate) * (np.log(epsilon) - np.log(x_line))



loss_diff_line = np.linspace(0, np.max(loss_diff), 100)
upper_bound3 = -1/np.log(1-learning_rate) * (np.log(loss_diff_line) + np.log(1/epsilon))

# plt.plot(x_line, y_line, linestyle='--', color='red', label='upper bound 1')
plt.plot(loss_diff_line, upper_bound3, linestyle='--', color='red', label='upper bound 3')
plt.legend()
plt.scatter(loss_diff, timesteps, color ='blue')
plt.xlabel('Distance of Starting Weight to Optimal Weight')
plt.ylabel('Iterations until convergence')
plt.title('Iterations until convergence given starting weight')
plt.show()

# print(distance_timestep_pairs)