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
# from tools import BlindColours, zero_balanced_weights
# from empiricalTest import LinearNetwork, get_random_regression_task
from linear_network import LinearNetwork
from utils import get_random_regression_task
# from scipy.linalg import expm
# from empiricalTest import QQT_new
import numpy as np
from utils import get_lambda_balanced
# from balanced_weights import balanced_weights
from scipy.linalg import qr


in_dim = 3
out_dim = 4
hidden_dim = 5

batch_size = 10

training_steps = 1000
learning_rate = 0.0001

lmda = 1

init_w1, init_w2 = get_lambda_balanced(1, in_dim, hidden_dim, out_dim)

# a = np.sqrt(lmda) + 1
# b = np.sqrt(2 * np.sqrt(lmda) + 1)

# init_w1 = np.eye(hidden_dim) * b
# init_w2 = np.eye(hidden_dim) * a

# init_w1, init_w2, _, q= balanced_weights(in_dim, hidden_dim, out_dim)

# while np.sign(q[0][0]) != np.sign(lmda):
#     init_w1, init_w2, _, q= balanced_weights(in_dim, hidden_dim, out_dim)

# init_w1 = init_w1 * np.sqrt(lmda/q[0][0])
# init_w2 = init_w2 * np.sqrt(lmda/q[0][0])


# init_w1, init_w2  = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.42)

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

##check the equation for derivative here, see if you can have F with negative evals as much as possible
##if we can have negative evals, then everything will be nice


#code to get F


#code to initialise linear network, get w1, w2s
model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

##
sigma_yx = 1/batch_size * Y @ X.T

#code for qqt_list
qqt_list = [np.vstack([
    np.hstack([w1.T @ w1, w1.T @ w2.T]),
    np.hstack([w2 @ w1, w2 @ w2.T])
]) for (w1, w2) in zip(w1s, w2s)]


def get_F(lmda):
    F = np.vstack([
        np.hstack([-lmda / 2 * np.eye(sigma_yx.shape[1]), sigma_yx.T]),
        np.hstack([sigma_yx, +lmda / 2 * np.eye(sigma_yx.shape[0])])
    ]) 
    return F 


def get_F_prime(lmda):
    F = np.vstack([
        np.hstack([lmda / 2 * np.eye(sigma_yx.shape[0]), sigma_yx]),
        np.hstack([sigma_yx.T, -lmda / 2 * np.eye(sigma_yx.shape[1])])
    ]) 
    return F 

F = get_F(lmda)
F_prime = get_F_prime(lmda)
#code for qqt_derivative list
qqt_derivatives = [(qqt_list[i+1] - qqt_list[i])/learning_rate for i in range(len(qqt_list)-1)]

#code for 
theoretical_deriv = [qqt @ F + F @ qqt - qqt @ qqt for qqt in qqt_list]
theoretical_deriv_prime = [qqt @ F_prime + F_prime @ qqt - qqt @ qqt for qqt in qqt_list]


diffs = [np.linalg.norm(emp - theo) for (emp, theo) in zip(qqt_derivatives, theoretical_deriv)]

print('hello')