import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import torch
# import torch.nn as nn 
# import torch.optim as optim

import matplotlib.pyplot as plt 
import numpy as np


class Aligned_Dynamics2:
    def __init__(self, init_w1, init_w2, X, Y):

        ldma = (init_w2.T @ init_w2 - init_w1 @ init_w1.T)[0][0]

        self.lmda = lmda

        sigma_yx_tilde = 1 / X.shape[1] * Y @ X.T 

        self.U, self.S, self.Vt = np.linalg.svd(sigma_yx_tilde)

        self.K = np.sqrt(lmda**2) + 4 * self.S**2 

        self.a0 = np.diag(self.U.T @ init_w2 @ init_w1 @ self.Vt.T) 

        self.C = ((self.K + lmda + 4 * self.S * self.a0 / np.sqrt(lmda**2 + 4*self.a0**2)) / 
        (self.K - lmda - 4 * self.S * self.a0 / np.sqrt(lmda**2 + 4*self.a0**2)))

        self.t = 0

    
    def forward(self, learning_rate):
        time_step = self.t * learning_rate

        K_exp = np.exp(self.K * time_step)

        x = (self.K * (self.C * K_exp -1) - self.lmda * (self.C * K_exp +1)) / (2*self.S * (self.C * K_exp +1))

        a_t = 2 * x / (1 - x**2)

        self.t += 1

        return a_t  
    

from utils import get_random_regression_task, get_lambda_balanced

in_dim = 3
hidden_dim = 3
out_dim = 3

batch_size = 3
learning_rate = 0.01
training_steps = 1000

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

X = X 
Y = Y
U, S, Vt = np.linalg.svd(1/batch_size*Y@X.T)

lmda = 0.01
init_w1_hat, init_w2_hat = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)

U_, S_, Vt_ = np.linalg.svd(init_w2_hat @ init_w1_hat)

init_w2 = U @ U_.T @ init_w2_hat  * 2
init_w1 = init_w1_hat @ Vt_.T @ Vt  * 2

# as_list = [np.diag(U.T @ w2 @ w1 @ Vt.T) for (w2, w1) in zip(w2s, w1s)]

# starting_svs = U.T @ init_w2 @ init_w1 
# theta0 = theta0 = np.arcsinh(starting_svs * 2 / lmda)
aligned_dynamics = Aligned_Dynamics2(init_w1, init_w2, X, Y)

analytical_dynamics = np.asarray([aligned_dynamics.forward(learning_rate) for _ in range(training_steps)])

print('hi')