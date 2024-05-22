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




##implement the solution for lambda balanced weights, see if it works


class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

##method for generating lambda balanced weights goes here

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

    sigma = 1

    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    w1 = np.random.randn(hidden_dim, in_dim)
    w2 = np.random.randn(out_dim, hidden_dim)

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

## class that computes dynamics goes here
#doing for equal input output dimensions (all the assumptions from Fukumizu)
class QQT_lambda_balanced:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]
        
        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

        U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T 

        self.F = np.vstack([
        np.hstack([self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
        np.hstack([sigma_yx_tilde, - self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ]) 

        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

        U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        self.U, self.S, self.V = U, S, Vt.T 

        self.F_inv = np.linalg.inv(self.F)

        self.O = 1/np.sqrt(2) * np.vstack([
            np.hstack([self.V_, self.V_]),
            np.hstack([self.U_, -self.U_])
        ])

        self.lmda_big = np.vstack([
            np.hstack([self.S_, np.zeros((self.S_.shape[0], self.S_.shape[0]))]),
            np.hstack([np.zeros((self.S_.shape[0], self.S_.shape[0])), -1. * self.S_])
        ])

        self.A = np.vstack([
        np.hstack([self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), np.zeros((sigma_yx_tilde.shape[1], sigma_yx_tilde.shape[1]))]),
        np.hstack([np.zeros((sigma_yx_tilde.shape[0], sigma_yx_tilde.shape[0])), - self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ]) 

        self.t = 0



    def forward(self, learning_rate):
        #performs forward for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        e_st_inv = np.diag(np.exp(-1. * np.diag(self.S_) * time_step))
        e_2st_inv = np.diag(np.exp(-2. * np.diag(self.S_) * time_step))

        Sinv = np.diag(1. / self.S)
        S_inv = np.diag(1. / np.diag(self.S_))

        B_t = np.exp(-self.lmda * time_step) * self.U.T @ self.U_ + self.V.T @ self.V_
        C_t = np.exp(-self.lmda * time_step) * self.U.T @ self.U_ - self.V.T @ self.V_
        
        B_t_inv = np.linalg.inv(B_t)

        # e_Ft_inv = (e -A) @ O @ (e -lmda) @ O.T 

        e_Ft_inv = expm(-1. * self.A * time_step) @ self.O @ expm(-1. * self.lmda_big * time_step) @ self.O.T
        # e_Ft_inv = expm(-1. * self.F * time_step)


        X_t = self.F_inv - e_Ft_inv @ self.F_inv @ e_Ft_inv

        e_lmda_inv = np.exp(-self.lmda * time_step)

        Z = np.vstack([
                self.V_ @ (i - e_st_inv @ C_t.T @ B_t_inv.T @ e_st_inv),
                self.U_ @ (i + e_st_inv @ C_t.T @ B_t_inv.T @ e_st_inv)
            ])
        
        center_left = 4. * e_st_inv @ B_t_inv @ Sinv @ B_t_inv.T @ e_st_inv 

        center_center = 0.5 * Z.T @ X_t @ Z 

        center = center_left + center_center 

        qqt = Z @ np.linalg.inv(center) @ Z.T

        ##CHOLESKY
        # L = np.linalg.cholesky(center)
        # y = np.linalg.solve(L, Z.T)
        # x = np.linalg.solve(L.T, y)
        # qqt = x.T @ Z.T

        
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 
    

class QQT_lambda_balanced2:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]

        

        
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

        U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        self.U, self.S, self.V = U, S, Vt.T

        self.S_inv = np.diag(1. / np.diag(self.S_))
 
        self.B = U.T @ U_ @ (i - self.lmda/2 * np.diag(self.S_inv)) + Vt @ V_ @ (i + self.lmda/2 * np.diag(self.S_inv))
        self.C = U.T @ U_ - Vt @ V_

        if np.isclose(np.linalg.det(self.B), 0):
            print('init_w1: ', init_w1)
            print('init_w2: ', init_w2)
            print('sigma_yx: ', sigma_yx_tilde)
            print('B: ', self.B)
            raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

        self.B_inv = np.linalg.inv(self.B)
        
        self.A_0 = S

        self.t = 0



    def forward(self, learning_rate):
        #performs forward for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        e_st_inv = np.diag(np.exp(-1. * np.diag(self.S_) * time_step))
        e_2st_inv = np.diag(np.exp(-2. * np.diag(self.S_) * time_step))

        S_inv = np.diag(1. / np.diag(self.S_))
        Sinv = np.diag(1./self.A_0)

        S_factor = np.diag(1/(np.diag(self.S_) + self.lmda**2/4*(1/np.diag(self.S_))))
        
        

        e_lmdat_inv = np.diag(np.exp(-1. * (np.diag(self.S_) + 1/4 * self.lmda**2 * np.diag(S_inv)) * time_step))
        e_2lmdat_inv = np.diag(np.exp(-2. * (np.diag(self.S_) + 1/4 * self.lmda**2 * np.diag(S_inv)) * time_step))

        Z = np.vstack([
            self.V_ @ ((i + self.lmda/2 *np.diag(S_inv)) - e_st_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv),
            self.U_ @ ((i - self.lmda/2 *np.diag(S_inv)) + e_st_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv),
        ])


        center_left = 4 * e_lmdat_inv @ self.B_inv @ Sinv @ self.B_inv.T @ e_lmdat_inv

        center_center = (i - e_2lmdat_inv) @ S_factor

        center_right = e_lmdat_inv @ self.B_inv @ self.C @ (e_2st_inv - i) @ S_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv

        center = center_left + center_center - center_right

        qqt = Z @ np.linalg.inv(center) @ Z.T

        ##CHOLESKY
        # L = np.linalg.cholesky(center)
        # y = np.linalg.solve(L, Z.T)
        # x = np.linalg.solve(L.T, y)
        # qqt = x.T @ Z.T

        
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 


## plot of dynamics goes here
in_dim = 5
hidden_dim = 5
out_dim = 5

batch_size = 10
learning_rate = 0.01
training_steps = 2

# init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

# init_w1, init_w2, _, _  = balanced_weights(in_dim, hidden_dim, out_dim)

lmda = 0.5
a = (lmda + 1)
b = np.sqrt(2*lmda + 1)

init_w1 = np.eye(hidden_dim)*a
init_w2 = np.eye(hidden_dim)*b

# init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

#lets try with lmda = 1


print((init_w1@init_w1.T - init_w2.T@init_w2)[0][0])


X, Y = get_random_regression_task(batch_size, in_dim, out_dim)


## model solution
model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

# analytical solution 1 (for comparision)
# analytical = QQT_new(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
# analytical = np.asarray([analytical.forward(learning_rate) for _ in range(training_steps)])

## analytical solution 2
analytical2 = QQT_lambda_balanced2(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
analytical2 = np.asarray([analytical2.forward(learning_rate) for _ in range(training_steps)])


# diffs = [np.linalg.norm(a - w) for (a, w) in zip(analytical, analytical2)]
# print(diffs[:20])
# print(diffs[-20:])
# first_a  = [a[0][0] for a in analytical]
# first_emp = [w[0][0][0] for w in ws]

# plt.figure()
# plt.plot(first_a, label = 'analytical')
# plt.plot(first_emp, label='empirical')
# plt.legend()
# plt.show()


# print(diffs)


### PLOTTING TRAJECTORIES OF THE REPRESENTATIONS
plot_items_n = 4
blind_colours = BlindColours().get_colours()


outputs = (np.asarray(ws)[:, 0, :, :] @ X[:,:plot_items_n])


for color, output in zip(blind_colours, outputs.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='output')
    plt.plot((analytical2 @ X[:,:plot_items_n]).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title('Learning Dynamics Lambda Balanced')
plt.xlabel('Training Steps')
plt.ylabel('Network Output')
plt.legend(['output', 'analytical'])

plt.show()