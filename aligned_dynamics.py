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
from empiricalTest import LinearNetwork, get_random_regression_task
from scipy.linalg import expm
from empiricalTest import QQT_new
import numpy as np
from balanced_weights import balanced_weights
from scipy.linalg import qr


class Aligned_Dynamics:
    def __init__(self, init_w1, init_w2, X, Y, theta0):
        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]

        self.theta0 = theta0

        sigma_yx_tilde = 1/ X.shape[1] * Y @ X.T

        self.U, self.S, self.Vt = np.linalg.svd(sigma_yx_tilde)

        self.K = np.sqrt(self.lmda**2 + 4 * self.S**2)

        self.c0 = (self.K + self.lmda + 2*self.S*np.tanh(self.theta0/2)) / (self.K - self.lmda - 2*self.S*np.tanh(self.theta0/2))
        
        self.t = 0
        
        
    def forward(self, learning_rate):
        time_step = self.t * learning_rate

        K_exp = np.exp(K * time_step)

        numerator = self.K * (self.c0 * K_exp - 1) - self.lmda * (self.c0 * K_exp+1)
        denominator = 2 * self.S * (self.c0 * K_exp + 1)

        theta = 2 * np.arctanh(numerator / denominator)

        a_t = self.lmda / 2 * np.sinh(theta)

        self.t += 1 

        return a_t 



## plot of dynamics goes here
    
#depending on dimensions you have different dynamics
in_dim = 5
hidden_dim = 5
out_dim = 5

batch_size = 5
learning_rate = 0.01
training_steps = 1000

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

U, S, Vt = np.linalg.svd(1/batch_size*Y@X.T)

# lmda = 0.5
# a = (np.sqrt(lmda) + 1)
# b = np.sqrt(2*np.sqrt(lmda) + 1)

# init_w1_hat = np.eye(hidden_dim)*a
# init_w2_hat = np.eye(hidden_dim)*b

# w1_out, w2_out, S_test, q

# init_w1_hat, init_w2_hat, _, lmda = balanced_weights(in_dim, hidden_dim, out_dim)

init_w1_hat, init_w2_hat, _, lmda = balanced_weights(hidden_dim, hidden_dim, hidden_dim)

lmda = np.abs(lmda[0][0])


H = np.random.randn(hidden_dim, hidden_dim)
R, _ = qr(H)

init_w1 = R @ init_w1_hat @ Vt 
init_w2 = U @ init_w2_hat @ R.T

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

ds_list = [np.diag(R.T @ w1 @ Vt.T) for w1 in w1s]
cs_list = [np.diag(U.T @ w2 @ R) for w2 in w2s]
as_list = [np.diag(U.T @ w2 @ w1 @ Vt.T) for (w2, w1) in zip(w2s, w1s)]
thetas_list = [np.arcsinh(2/lmda * a) for a in as_list]

theta0 = thetas_list[0]

K = np.sqrt(lmda**2 + 4*S**2)




lmdas = [(ds**2 - cs**2) / lmda for (ds, cs) in zip(ds_list, cs_list)]

tau_dadt = np.diff(as_list, axis=0) / learning_rate
tau_dddt = np.diff(ds_list, axis=0) / learning_rate
tau_dcdt = np.diff(cs_list, axis = 0) / learning_rate
tau_dthetadt = np.diff(thetas_list, axis = 0) / learning_rate

exp0 = [np.sqrt(lmda**2+ 4*a**2) * (S - a) for a in as_list]
exp1 = [d * (S - c*d) for (c, d) in zip(cs_list, ds_list)]
exp2 = [c * (S - c*d) for (c, d) in zip(cs_list, ds_list)]
exp3 = [2 * S - lmda * np.sinh(theta) for theta in thetas_list]

singular_vals = [np.diag(U.T @ w2 @ w1 @ Vt.T) for (w2, w1) in zip(w2s, w1s)]

##making substitution a = lambda / 2 sinh(theta)

theta0 = np.arcsinh(singular_vals[0] * 2 / lmda)


aligned_dynamics = Aligned_Dynamics(init_w1, init_w2, X, Y, theta0=theta0)

analytical_dynamics = [aligned_dynamics.forward(learning_rate) for _ in range(training_steps)]




singular_vals = np.array(singular_vals).T
print(singular_vals)

plt.figure(figsize=(10, 6))  # Set the figure size
for i, theta in enumerate(singular_vals, start=1):
    plt.plot(theta, label=f'Theta for Singular Value {i}')

plt.xlabel('Time / Iteration')
plt.ylabel('Theta for Singular Value')
plt.title('Evolution of Singular Values Over Time')
plt.legend()
plt.grid(True)
plt.show()


analytical_dynamics = np.array(analytical_dynamics).T
plt.figure(figsize=(10, 6))  # Set the figure size
for i, theta in enumerate(analytical_dynamics, start=1):
    plt.plot(theta, label=f'Theta for Singular Value {i}')

plt.xlabel('Time / Iteration')
plt.ylabel('Analytical Theta for Singular Value')
plt.title('Evolution of Singular Values Over Time')
plt.legend()
plt.grid(True)
plt.show()
# def plot_singular_val(n):
#     if n >= len(singular_vals[0]):
#         print('not enough singular values')
#         return 
    
#     singular_val = [sv[n] for sv in singular_vals]

#     plt.title('Evolution of Singular Value: ' + str(n))
#     plt.axhline(y=S[n], label='true singular value', color='red')
#     plt.plot(singular_val, label='trajectory', color='blue')
#     plt.legend()
#     plt.show()


# plot_singular_val(0)