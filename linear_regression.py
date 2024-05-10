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


##LINEAR REGRESSION IN 1D CASE###

#write up code for this, write up code for linear regression
in_dim = 1
hidden_dim = 1
out_dim = 1

batch_size = 10
learning_rate = 0.1
training_steps = 400

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)
init_w1 = np.abs(init_w1)
init_w2 = np.abs(init_w2)


model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1.copy(), init_w2=init_w2.copy())


num_tasks = 2

tasks = [get_random_regression_task(batch_size, in_dim, out_dim) for _ in range(num_tasks)]
tasks = [(np.abs(task[0]), np.abs(task[1])) for task in tasks]
##special step for 1D weights to ensure self.B is non singular

w1s_list = []
w2s_list = []


for (X, Y) in tasks:
    w1s, w2s, _ = model.train(X, Y, training_steps, learning_rate)
    w1s_list.extend(w1s)
    w2s_list.extend(w2s)


analyticals_list = []


analytical = None 
for i, (X, Y) in enumerate(tasks):
    if analytical is None:
        analytical = QQT_new(init_w1, init_w2, X.T, Y.T ,True)
    else:
        analytical = QQT_new(w1s_list[training_steps*i], w2s_list[training_steps*i], X.T, Y.T, True)
    analytical = np.asarray([analytical.forward(learning_rate) for _ in range(training_steps)])
    analyticals_list.extend(analytical)

losses_list = []
w2w1s_list = [w2 @ w1 for (w2, w1) in zip(w2s_list, w1s_list)]

for i, (X, Y) in enumerate(tasks):
    output_empirical = [w2w1 @ X for w2w1 in w2w1s_list]
    losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_empirical[training_steps*i:training_steps*(i+1)]]
    losses_list.extend(losses)

anal_losses_list = []
for i, (X, Y) in enumerate(tasks):
    output_analytical = [a @ X for a in analyticals_list]
    losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_analytical[training_steps*i:training_steps*(i+1)]]
    anal_losses_list.extend(losses)


joined_X = np.hstack([tasks[0][0], tasks[1][0]])
joined_Y = np.hstack([tasks[0][1], tasks[1][1]])


w2w1_opt = np.linalg.inv(joined_X@ joined_X.T) @ joined_X @ joined_Y.T

opt_losses_list = []
for i, (X, Y) in enumerate(tasks):
    output_opt = [w2w1_opt @ X for _ in range(training_steps)]
    losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_opt]
    opt_losses_list.extend(losses)


print('helloe')
plt.figure()

plt.plot(anal_losses_list, label = 'analytical losses')
plt.plot(losses_list, label = 'empirical losses')
plt.plot(opt_losses_list, label = 'loss optimal')
plt.title('Loss dynamics in 1D regression task')
plt.legend()
plt.show()




##LINEAR REGRESSION USING SVD##


