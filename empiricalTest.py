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

class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

class QQT_new:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):
        
        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

        U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T 

        self.dim_diff = np.abs(self.input_dim - self.output_dim)

        if self.input_dim < self.output_dim:
            U_hat = U_[:, self.input_dim:]
            V_hat = np.zeros((self.input_dim, self.dim_diff))
            U_ = U_[:, :self.output_dim]

        elif self.input_dim > self.output_dim:
            U_hat = np.zeros((self.output_dim, self.dim_diff))
            V_hat = V_[:, self.output_dim:]
            V_ = V_[:, :self.output_dim]

        else:
            U_hat  = None 
            V_hat = None

        self.U_hat = U_hat 
        self.V_hat = V_hat 
        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

        U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        self.U, self.S, self.V = U, S, Vt.T 

        self.B = self.U.T @ self.U_ + self.V.T @ self.V_ 
        self.C = self.U.T @ self.U_ - self.V.T @ self.V_


        

        ##CHECK THAT B IS NON SINGULAR

        if np.isclose(np.linalg.det(self.B), 0):
            print('init_w1: ', init_w1)
            print('init_w2: ', init_w2)
            print('sigma_yx: ', sigma_yx_tilde)
            print('B: ', self.B)
            raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")
        
        self.t = 0

    def forward(self, learning_rate):
        #performs forward for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        e_st_inv = np.diag(np.exp(-1. * np.diag(self.S_) * time_step))
        e_2st_inv = np.diag(np.exp(-2. * np.diag(self.S_) * time_step))

        B_inv = np.linalg.inv(self.B)

        Sinv = np.diag(1. / self.S)
        S_inv = np.diag(1. / np.diag(self.S_))

        # if self.t==10:
        #     print('QQT changed version:')
        #     print('V_: ', self.V_)
        #     print('U_: ', self.U_)
        #     print('e_st_inv: ', e_st_inv)
        #     print('C: ', self.C)
        #     print('B_inv: ', B_inv)
        #     print('V_hat: ', self.V_hat)
        #     print('V:  ', self.V)
        #     print('U_hat: ', self.U_hat)
        #     print('U:  ', self.U)

        if self.U_hat is None and self.V_hat is None:
            Z = np.vstack([
                self.V_ @ (i - e_st_inv @ self.C.T @ B_inv.T @ e_st_inv),
                self.U_ @ (i + e_st_inv @ self.C.T @ B_inv.T @ e_st_inv)
            ])
            center_right = 0.

        else:
            Z = np.vstack([
                self.V_ @ (i - e_st_inv @ self.C.T @ B_inv.T @ e_st_inv) + 2*self.V_hat@self.V_hat.T @ self.V @ B_inv.T @ e_st_inv,
                self.U_ @ (i + e_st_inv @ self.C.T @ B_inv.T @ e_st_inv) + 2*self.U_hat@self.U_hat.T @ self.U @ B_inv.T @ e_st_inv
            ])
            center_right = 4 * time_step * e_st_inv @ B_inv @ (self.V.T @ self.V_hat @ self.V_hat.T @ self.V + self.U.T @ self.U_hat @ self.U_hat.T @ self.U) @ B_inv.T @ e_st_inv

        center_left = 4. * e_st_inv @ B_inv @ Sinv @ B_inv.T @ e_st_inv 
        center_center = (i - e_2st_inv) @ S_inv- e_st_inv @ B_inv @ self.C @ (e_2st_inv - i) @ S_inv @ self.C.T @ B_inv.T @ e_st_inv
        
        # if self.t==1:
        #     print('New Version center_center variables: ')
        #     print('i: ', i)
        #     print('e_2st_inv: ', e_2st_inv)
        #     print('S_inv: ', S_inv)
        #     print('e_st_inv: ', e_st_inv)
        #     print('B_inv: ', B_inv)
        #     print('C: ', self.C)

        center = center_left + center_center + center_right

        #TODO: cholesky decomposition, QR decomposition

        #Zt X-1 Z
        # X = Lt L using Qr or cholesky
        # Ly = Z, y = L-1 Z
        #yty
        

        # qqt = Z @ np.linalg.inv(center) @ Z.T 

        ##CHOLESKY
        L = np.linalg.cholesky(center)
        y = np.linalg.solve(L, Z.T)
        x = np.linalg.solve(L.T, y)
        qqt = x.T @ Z.T

        ##QR decomposition
        Q, R = np.linalg.qr(center)
        QTZ = Q.T @ Z .T
        x = np.linalg.solve(R, QTZ)
        qqt2 = x.T @ Z.T
        # x = Q.T @ y 
        # qqt2 = x.T @ x

        qqt3 = Z @ np.linalg.inv(center) @ Z.T 
        
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 

class LinearNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1 = None, init_w2 = None):

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x): 
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):

        # self.forward(x)

        forward = self.W2 @ self.W1 @ x
        dW1 = 1/x.shape[1] * self.W2.T @ (forward-y) @ x.T 
        dW2 = 1/x.shape[1] * (forward - y) @ x.T @ self.W1.T 


        # dW1 = 1/x.shape[1] * self.W2.T @ (self.z-y) @ x.T 
        # dW2 = 1/x.shape[1] * (self.z - y) @ x.T @ ###.T

        # print('dW1: ', np.linalg.norm(dW1))
        # print('dW2: ', np.linalg.norm(dW2))

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1


    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        # print('X_train shape: ', X_train.shape)
        # print('Y_train shape: ', Y_train.shape)
        losses = []
        # print('epochs: ', epochs)
        for _ in range(epochs):
            # loss = 1/(2*X_train.shape[1]) * np.linalg.norm((W))
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        return w1s, w2s, losses
    
def whiten(X):

    scaler = StandardScaler()

    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    # print('explained variance: ', pca.explained_variance_)
    X_whitened = X_pca / np.sqrt(pca.explained_variance_)

    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = np.random.randn(batch_size, in_dim)
    Y = np.random.randn(batch_size, out_dim)
    X_whitened = whiten(X)

    return X_whitened.T, Y.T




####TESTING####

in_dim = 5
hidden_dim = 10
out_dim = 2

batch_size = 10
learning_rate = 0.1
training_steps = 400

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)



from dynamics import QQT 
## analytical solution
analytical = QQT_new(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
analytical = np.asarray([analytical.forward(learning_rate) for _ in range(training_steps)])

## model solution
model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

# diffs = [np.linalg.norm(a - w) for (a, w) in zip(analytical, ws)]

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
    plt.plot((analytical @ X[:,:plot_items_n]).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title('Learning Dynamics Empirical vs Analytical')
plt.xlabel('Training Steps')
plt.ylabel('Network Output')
plt.legend(['output', 'analytical'])

plt.show()


# plt.set_xlim(1, training_steps * 1.6)
# plt.set_ylim([-2.2, 2.2])

# plt.set_xscale("log")

# plt.set_xlabel("Training Steps")
# if i == 0:
#     sns.despine(ax=plt)
#     plt.set_ylabel("Network Output")
#     axs[i].set_yticks([-2, -1., 0., 1., 2])
# else:
#     sns.despine(ax=axs[i], left=True)
#     axs[i].set_yticks([])



##expression for loss analytical
"""
We have three tasks: $T_i, T_j, T_k$ which we are learning in alphabetical order. 
We define $F_i(T_j, T_k)$ as how much we forget about $T_i$ while learning $T_k$, compared 
to how much we knew after having just learned $T_j$
"""

# in_dim = 5
# hidden_dim = 10
# out_dim = 2

# batch_size = 10
# learning_rate = 0.1
# training_steps = 400



# init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)
# model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1.copy(), init_w2=init_w2.copy())

# num_tasks = 3

# tasks = [get_random_regression_task(batch_size, in_dim, out_dim) for _ in range(num_tasks)]

# w1s_list = []
# w2s_list = []

# for (X, Y) in tasks:
#     w1s, w2s, _ = model.train(X.T, Y.T, training_steps, learning_rate)
#     w1s_list.extend(w1s)
#     w2s_list.extend(w2s)


# analyticals_list = []

# analytical = None 
# for i, (X, Y) in enumerate(tasks):
#     if analytical is None:
#         analytical = QQT_new(init_w1, init_w2, X, Y ,True)
#     else:
#         analytical = QQT_new(w1s_list[training_steps*i], w2s_list[training_steps*i], X, Y, True)
#     analytical = np.asarray([analytical.forward(learning_rate) for _ in range(training_steps)])
#     analyticals_list.extend(analytical)


# losses_list = []
# w2w1s_list = [w2 @ w1 for (w2, w1) in zip(w2s_list, w1s_list)]

# for i, (X, Y) in enumerate(tasks):
#     output_empirical = [X @ w2w1.T for w2w1 in w2w1s_list]
#     losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_empirical]
#     losses_list.append(losses)

# anal_losses_list = []
# for i, (X, Y) in enumerate(tasks):
#     output_analytical = [X @ a.T for a in analyticals_list]
#     losses = [0.5 * 1 / batch_size * np.linalg.norm(out - Y, ord='fro')**2 for out in output_analytical]
#     anal_losses_list.append(losses)



# print('hi')

# ###LOSS ANALYTICAL VS EMPIRICAL###
# first_loss = losses_list[0]

# (X_i, Y_i) = tasks[0]
# sigma_yx_i =  Y_i.T @ X_i
# sigma_yy_i = Y_i @ Y_i.T

# analytical_loss = [0.5 * 1 / batch_size * (np.linalg.norm(w - sigma_yx_i)**2
#                                            -np.trace(sigma_yx_i @ sigma_yx_i.T)
#                                            + np.trace(sigma_yy_i)) for w in analytical]


# ###FORGETTING ANALYTICAL VS EMPIRICAL### 
# forgetting_empirical = losses_list[0][2*training_steps:] - losses_list[0][2*training_steps]

# plt.plot(forgetting_empirical, label='empirical forgetting')
# plt.legend()
# plt.title('Empirical Forgetting')
# plt.show()


# ###RATE OF LOSS###


# ###RATE OF FORGETTING###

# ###LINEAR REGRESSION 1D CASE##3


