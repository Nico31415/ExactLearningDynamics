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



"""
We have three tasks: $T_i, T_j, T_k$ which we are learning in alphabetical order. 
We define $F_i(T_j, T_k)$ as how much we forget about $T_i$ while learning $T_k$, compared 
to how much we knew after having just learned $T_j$
"""

in_dim = 5
hidden_dim = 10
out_dim = 2

batch_size = 10
learning_rate = 0.001
training_steps = 400



init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)
model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1.copy(), init_w2=init_w2.copy())

num_tasks = 3

tasks = [get_random_regression_task(batch_size, in_dim, out_dim) for _ in range(num_tasks)]

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
    losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_empirical]
    losses_list.append(losses)

anal_losses_list = []
for i, (X, Y) in enumerate(tasks):
    output_analytical = [a @ X for a in analyticals_list]
    losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_analytical]
    anal_losses_list.append(losses)



print('hi')

###LOSS ANALYTICAL VS EMPIRICAL###
first_loss = losses_list[0]

(X_i, Y_i) = tasks[0]
sigma_yx_i =  Y_i@ X_i.T
sigma_yy_i = Y_i @ Y_i.T

# losses_v0 = [0.5 * 1 / batch_size * np.trace((X_i @sigma_j.T - Y_i) @ (X_i @ sigma_j.T - Y_i).T) for sigma_j in analyticals_list]

analytical_loss = [0.5 * 1 / batch_size * (np.linalg.norm(np.sqrt(batch_size) * sigma_j - 1/np.sqrt(batch_size) * sigma_yx_i, ord='fro')**2
                                     - 1/batch_size * np.trace(sigma_yx_i @ sigma_yx_i.T)
                                     + np.trace(sigma_yy_i)) for sigma_j in analyticals_list]


plt.figure()
plt.plot(analytical_loss, label='analytical loss')
plt.plot(first_loss, label='empirical')
plt.title('Empirical vs Analytical Loss in terms of correlations')
plt.legend()
plt.show()



###FORGETTING ANALYTICAL VS EMPIRICAL### 
forgetting_empirical = losses_list[0][2*training_steps:] - losses_list[0][2*training_steps]

sigma_k = analyticals_list[2*training_steps]
forgetting_analytical = [0.5 * (np.linalg.norm(sigma_j - 1/batch_size * sigma_yx_i)**2 - np.linalg.norm(sigma_k - 1/batch_size * sigma_yx_i)**2) for sigma_j in analyticals_list[2*training_steps:]]

plt.figure()
plt.plot(forgetting_analytical, label='analytical forgetting')
plt.plot(forgetting_empirical, label='empirical')
plt.title('Empirical vs Analytical Forgetting in terms of correlations')
plt.legend()
plt.show()

###RATE OF LOSS###
def calculate_derivative(time_series):
    derivative = [time_series[i+1] - time_series[i] for i in range(len(time_series)-1)]
    return derivative


loss_rate = calculate_derivative(first_loss)
analyticals_list_rate = calculate_derivative(analyticals_list)

loss_rate_anaytical = [np.trace(sigma_j @ sigma_j_prime.T) - 1/batch_size * np.trace(sigma_j_prime @ sigma_yx_i.T) 
                       for (sigma_j, sigma_j_prime) in zip(analyticals_list, analyticals_list_rate)]

plt.figure()
plt.plot(loss_rate_anaytical, label='analytical loss rate')
plt.plot(loss_rate, label='empirical')
plt.title('Empirical vs Analytical Loss Rate in terms of correlations')
plt.legend()
plt.show()

###DO IT IN A MORE COMPLICATED WAY###
F = np.vstack([np.hstack([np.zeros((sigma_yx_i.shape[1], sigma_yx_i.shape[1])), sigma_yx_i.T]),
               np.hstack([sigma_yx_i, np.zeros((sigma_yx_i.shape[0], sigma_yx_i.shape[0]))])])

qqts_list = []
qqts = None 
for i, (X, Y) in enumerate(tasks):
    if qqts is None:
        qqts = QQT_new(init_w1, init_w2, X.T, Y.T ,False)
    else:
        qqts = QQT_new(w1s_list[training_steps*i], w2s_list[training_steps*i], X.T, Y.T, False)
    qqts = np.asarray([qqts.forward(learning_rate) for _ in range(training_steps)])
    qqts_list.extend(qqts)

qqt_primes = [learning_rate * (F @ qqt + qqt @ F - qqt @ qqt.T) for qqt in qqts_list] 

w2w1s_primes = [qqt_prime[in_dim:, :in_dim] for qqt_prime in qqt_primes]

loss_rate_anaytical2 = [np.trace(sigma_j @ sigma_j_prime.T) - 1/batch_size * np.trace(sigma_j_prime @ sigma_yx_i.T) 
                       for (sigma_j, sigma_j_prime) in zip(analyticals_list, w2w1s_primes)]
plt.figure()
plt.plot(loss_rate_anaytical2, label='analytical loss rate 2')
plt.plot(loss_rate, label='empirical')
plt.title('Empirical vs Analytical Loss Rate in terms of correlations 2')
plt.legend()
plt.show()




###RATE OF FORGETTING###
forgetting_rate = calculate_derivative(forgetting_empirical)
analyticals_list_rate = calculate_derivative(analyticals_list)

forgetting_rate_anaytical = [np.trace(sigma_j @ sigma_j_prime.T) - 1/batch_size * np.trace(sigma_j_prime @ sigma_yx_i.T) 
                       for (sigma_j, sigma_j_prime) in zip(analyticals_list, analyticals_list_rate)]

plt.figure()
plt.plot(loss_rate_anaytical, label='analytical forgetting rate')
plt.plot(loss_rate, label='empirical')
plt.title('Empirical vs Analytical Forgetting Rate in terms of correlations')
plt.legend()
plt.show()

F = np.vstack([np.hstack([np.zeros((sigma_yx_i.shape[1], sigma_yx_i.shape[1])), sigma_yx_i.T]),
               np.hstack([sigma_yx_i, np.zeros((sigma_yx_i.shape[0], sigma_yx_i.shape[0]))])])

qqts_list = []
qqts = None 
for i, (X, Y) in enumerate(tasks):
    if qqts is None:
        qqts = QQT_new(init_w1, init_w2, X.T, Y.T ,False)
    else:
        qqts = QQT_new(w1s_list[training_steps*i], w2s_list[training_steps*i], X.T, Y.T, False)
    qqts = np.asarray([qqts.forward(learning_rate) for _ in range(training_steps)])
    qqts_list.extend(qqts)

qqt_primes = [1/learning_rate* (F @ qqt + qqt @ F - qqt @ qqt.T) for qqt in qqts_list] 

w2w1s_primes = [qqt_prime[in_dim:, :in_dim] for qqt_prime in qqt_primes]

loss_rate_anaytical2 = [np.trace(sigma_j @ sigma_j_prime.T) - 1/batch_size * np.trace(sigma_j_prime @ sigma_yx_i.T) 
                       for (sigma_j, sigma_j_prime) in zip(analyticals_list, w2w1s_primes)]
plt.figure()
plt.plot(loss_rate_anaytical2, label='analytical forgetting rate 2')
plt.plot(forgetting_rate, label='empirical')
plt.title('Empirical vs Analytical Forgetting Rate in terms of correlations 2')
plt.legend()
plt.show()

###LINEAR REGRESSION 1D CASE##3