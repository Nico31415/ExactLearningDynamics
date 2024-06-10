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
# from scipy.linalg import expm
# from empiricalTest import QQT_new
# from balanced_weights import balanced_weights


from utils import get_lambda_balanced, get_random_regression_task
from linear_network import LinearNetwork
class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

class QQT_lambda_balanced3:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

        self.lmda = (init_w2.T @ init_w2 - init_w1 @ init_w1.T)[0][0] 

        

        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]



        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 
        
        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

        U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T 

        self.F = np.vstack([
        np.hstack([- self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
        np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ]) 

        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_
        
        self.dim_diff = np.abs(self.input_dim - self.output_dim)

        if self.input_dim < self.output_dim:
            U_hat = U_[:, self.input_dim:]
            V_hat = np.zeros((self.input_dim, self.dim_diff))
            U_ = U_[:, :self.input_dim]

        elif self.input_dim > self.output_dim:
            U_hat = np.zeros((self.output_dim, self.dim_diff))
            V_hat = V_[:, self.output_dim:]
            V_ = V_[:, :self.output_dim]
        
        else:
            U_hat = None
            V_hat = None 

        self.U_hat, self.V_hat = U_hat, V_hat
        self.U_, self.V_ = U_, V_

        # U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        U, S, Vt  = np.linalg.svd(init_w2 @ init_w1)
        self.U, self.S, self.V = U, S, Vt.T

        self.S_inv = np.diag(1. / np.diag(self.S_))

        self.S_ = np.diag(self.S_)

        self.X = (np.sqrt(self.lmda**2 + 4*self.S_**2) - 2 * self.S_)/self.lmda
        self.A = 1 / (np.sqrt(1 + self.X**2))

        self.X = np.diag(self.X)
        self.A = np.diag(self.A)


        self.S2 = np.sqrt((self.lmda + np.sqrt(self.lmda**2 + 4*self.S**2)) / 2)

        # _ , self.S2, _ = np.linalg.svd(init_w2)
        # _ , self.S1, _ = np.linalg.svd(init_w2)

        # S2_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) + lmda) / 2))
        # S1_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) - lmda) / 2))

        # if out_dim > in_dim:
        #     add_terms = np.asarray([np.sqrt(lmda) for _ in range(hidden_dim - in_dim)])
        #     S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
        #                     np.zeros((hidden_dim - in_dim, hidden_dim))]) 
        #     S1 = np.vstack([np.diag(S1_equal_dim), 
        #                     np.zeros((hidden_dim - in_dim, in_dim))])
        # elif in_dim > out_dim:
        #     add_terms = np.asarray([-np.sqrt(-lmda) for _ in range(hidden_dim-out_dim)])
        #     S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
        #                     np.zeros((hidden_dim, in_dim - hidden_dim))])
        #     S2 = np.hstack([np.diag(S2_equal_dim), 
        #                     np.zeros((out_dim, hidden_dim - out_dim))]) 
        
        # self.S2 = S2 
        # self.S1 = S1
        
        # self.S1 = self.S / self.S2
        # self.S2 = np.diag(self.S2)
        # self.S1 = np.diag(self.S1)

        _, s1, _ = np.linalg.svd(init_w1)
        _, s2, _ = np.linalg.svd(init_w2)

        self.S1 = np.zeros((hidden_dim, in_dim))
        self.S1[:len(s1), :len(s1)] = np.diag(s1)

        self.S2 = np.zeros((out_dim, hidden_dim))
        self.S2[:len(s2), :len(s2)] = np.diag(s2)

        self.B = self.S2.T @ U.T @ U_ @ (self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A - self.X @ self.A)
        self.C = self.S2.T @ U.T @ U_  @ (self.A - self.X @ self.A) - self.S1 @ Vt @ V_ @ (self.X @ self.A + self.A)

        self.sign = np.sign(self.output_dim-self.input_dim)

        self.eval = np.sqrt(self.S_**2 + self.lmda**2 / 4)
        self.eval_inv = np.diag(1. /self.eval)
        self.eval_extra_dim = self.sign * self.lmda/2 * np.ones(self.dim_diff)
        self.eval_extra_dim_inv = np.diag(1/self.eval_extra_dim)

        # if np.isclose(np.linalg.det(self.B), 0):
        #     print('init_w1: ', init_w1)
        #     print('init_w2: ', init_w2)
        #     print('sigma_yx: ', sigma_yx_tilde)
        #     print('B: ', self.B)
        #     raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

        # self.B_inv = np.linalg.inv(self.B)
        
        self.B_inv = np.linalg.pinv(self.B)

        

        
        # self.A_0 = S

        self.t = 0



    def forward(self, learning_rate):
        #performs forward for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        e_eval_st_inv  = np.diag(np.exp(-1. * self.eval * time_step))
        e_eval_2st_inv  = np.diag(np.exp(-2. * self.eval * time_step))
        e_eval_st_extra_dim = np.diag(np.exp(1. * self.eval_extra_dim * time_step))
        e_eval_2st_extra_dim = np.diag(np.exp(2. * self.eval_extra_dim * time_step))

        # Sinv = np.diag(1./self.A_0)


        if self.U_hat is None and self.V_hat is None:
            Z = np.vstack([
                self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
                self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv)
            ])
            center_add = 0.

        else:
            Z = np.vstack([
                self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv) + 2 * self.V_hat @ e_eval_st_extra_dim @ self.V_hat.T @ self.V @ self.S1.T @ self.B_inv.T @ e_eval_st_inv,
                self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv) + 2 * self.U_hat @ e_eval_st_extra_dim @ self.U_hat.T @ self.U @ self.S2 @ self.B_inv.T @ e_eval_st_inv
            ])

            min_dim = min(in_dim, hidden_dim)
            s1_add = self.S1[:min_dim, :min_dim]
            s2_add = self.S2[:min_dim, :min_dim]
            # center_add = np.sqrt(2) * (self.S1 @ self.V.T @ self.V_hat + self.S2.T @ self.U.T @ self.U_hat) @ (e_eval_2st_extra_dim - np.eye(self.dim_diff)) @ self.eval_extra_dim_inv @ (self.V_hat.T @ self.V @ self.S1.T + self.U_hat.T @ self.U @ self.S2)

            center_add = np.sqrt(2) * (s1_add @ self.V.T @ self.V_hat + s2_add @ self.U.T @ self.U_hat) @ (e_eval_2st_extra_dim - np.eye(self.dim_diff)) @ self.eval_extra_dim_inv @ (self.V_hat.T @ self.V @ s1_add + self.U_hat.T @ self.U @ s2_add)

        
        center_left = 4 * e_eval_st_inv @ self.B_inv @ self.B_inv.T @ e_eval_st_inv

        # center_left = 4 * e_eval_st_inv @ self.B_inv @ Sinv @ self.B_inv.T @ e_eval_st_inv

        center_center = (i - e_eval_2st_inv) @ self.eval_inv

        center_right = e_eval_st_inv @ self.B_inv @ self.C @ (e_eval_2st_inv - i) @ self.eval_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv

        center = center_left + center_center - center_right + center_add

        # qqt = Z @ np.linalg.inv(center) @ Z.T

        #CHOLESKY
        L = np.linalg.cholesky(center)
        y = np.linalg.solve(L, Z.T)
        x = np.linalg.solve(L.T, y)
        qqt = x.T @ Z.T

        # # add_term = np.diag([self.lmda for _ in range(self.input_dim)] + [-self.lmda for _ in range(self.input_dim)])

        # qqt = qqt + add_term

        
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 
    



in_dim = 3
hidden_dim = 5
out_dim = 7

lmda = 1

batch_size = 10
learning_rate = 0.001 / lmda
training_steps = int(200 * np.sqrt(lmda))

# init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

init_w1, init_w2  = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)
# lmda = lmda[0][0]


# a = (np.sqrt(lmda) + 1)
# b = np.sqrt(2*np.sqrt(lmda) + 1)

# init_w1 = np.eye(hidden_dim)*a
# init_w2 = np.eye(hidden_dim)*b

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

U_, S_, Vt_ = np.linalg.svd(Y @ X.T / batch_size)

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

analytical2 = QQT_lambda_balanced3(init_w1.copy(), init_w2.copy(), X.T, Y.T, False)
analytical2 = np.asarray([analytical2.forward(learning_rate) for _ in range(training_steps)])

rep1 = [[w1.T @ w1] for w1 in w1s]
rep1_analytical = np.array([a[:in_dim, :in_dim] for a in analytical2])

rep2= [[w2 @ w2.T] for w2 in w2s]
rep2_analytical = np.array([a[in_dim:, in_dim:] for a in analytical2])

reps = (np.asarray(rep2)[:, 0, :, :])

plot_items_n = 4
blind_colours = BlindColours().get_colours()


outputs = (np.asarray(ws)[:, 0, :, :] @ X[:,:plot_items_n])

# representations = (np.asarray(w1s))

plt.figure()
reps = (np.asarray(rep2)[:, 0, :, :])
for color, output in zip(blind_colours, reps.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='Representation')
    plt.plot((rep2_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Representation (W2)')
plt.legend(['output', 'analytical'])


plt.figure()
reps = (np.asarray(rep1)[:, 0, :, :])
for color, output in zip(blind_colours, reps.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='Representation')
    plt.plot((rep1_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Representation (W1)')
plt.legend(['output', 'analytical'])




plt.figure()

analytical2 = [a[in_dim:, :in_dim] for a in analytical2]
for color, output in zip(blind_colours, outputs.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='output')
    plt.plot((analytical2 @ X[:,:plot_items_n]).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Learning Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Output')
plt.legend(['output', 'analytical'])

plt.show()