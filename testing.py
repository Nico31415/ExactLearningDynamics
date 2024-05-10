import torch
import torch.nn as nn 
import torch.optim as optim

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import seaborn as sns
from tools import BlindColours, zero_balanced_weights




class LinearNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1 = None, init_w2 = None):

        if init_w1 is not None and init_w2 is not None:

            self.W1  = init_w1
            self.W2 = init_w2
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x):
        
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):

        self.forward(x)

        dW1 =  1/x.shape[1] *self.W2.T @ (self.z-y) @ x.T 
        dW2 = 1/x.shape[1] * (self.z - y) @ x.T @ self.W1.T

        self.W2 -= learning_rate * dW2 
        self.W1 -= learning_rate * dW1 


    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        print('X_train shape: ', X_train.shape)
        print('Y_train shape: ', Y_train.shape)
        losses = []
        loss = np.mean((self.forward(X_train) - Y_train) ** 2)
        losses.append(loss)
        w1s.append(self.W1)
        w2s.append(self.W2)
        for _ in range(epochs):
            
            self.backward(X_train, Y_train, learning_rate)
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1)
            w2s.append(self.W2)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        return w1s, w2s, losses
    
class QQT:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):
        
        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        sigma_yx_tilde = 1 / self.batch_size * Y.T@ X 

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

        center = np.linalg.inv(center_left + center_center + center_right)

        qqt = Z @ center @ Z.T 
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 
    

def whiten(X):

    scaler = StandardScaler()

    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    print('explained variance: ', pca.explained_variance_)
    X_whitened = X_pca / np.sqrt(pca.explained_variance_)

    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = np.random.randn(batch_size, in_dim)
    Y = np.random.randn(batch_size, out_dim)
    X_whitened = whiten(X)

    return X_whitened, Y


in_dim = 5
hidden_dim = 10
out_dim = 2

tau = 0.1
batch_size = 10
epochs = 150

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
# init_w1 = torch.tensor(init_w1)
# init_w2 = torch.tensor(init_w2)

model = LinearNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, init_w1=init_w1, init_w2=init_w2)
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=tau)



X_train, Y_train = get_random_regression_task(batch_size, in_dim, out_dim)

print('model: ', type(model))
print('X: ', X_train)
print('Y: ', Y_train)
print('epochs: ', type(epochs))


w1s, w2s, losses = model.train(X_train.T, Y_train.T, epochs, tau)


# analytical = QQT(init_w1, init_w2, X_train, Y_train, True)
analytical = QQT(init_w1, init_w2, X_train, Y_train, True)
analytical = np.asarray([analytical.forward(tau) for _ in range(epochs)])

w2w1s = [w2 @ w1 for (w2, w1) in zip(w2s, w1s)]
diffs = [np.linalg.norm(w2w1 - anal) for (w2w1, anal) in zip(w2w1s, analytical)]

plt.figure()
plt.plot(diffs)

top_left_empirical = [w2w1[0][0] for w2w1 in w2w1s]
top_left_analytical = [anal[0][0] for anal in analytical]



#TODO plot trajectories of N items
plt.plot(top_left_analytical, label='analytical')
plt.plot(top_left_empirical, label='empirical')
plt.title('Trajectories of top left item of both representations')
plt.legend()
plt.show()

anal_loss = np.linalg.norm(X_train @ (analytical[-1]).T - Y_train)
emp_loss = np.linalg.norm(X_train @ (w2w1s[-1]).T - Y_train)



output_analytical = [(X_train) @ (anal).T for anal in analytical]
output_empirical = [(X_train) @ (w2w1).T for w2w1 in w2w1s]

losses_empirical = [0.5 * out_dim / batch_size * np.linalg.norm(out - Y_train)**2 for out in output_empirical]
losses_analytical = [0.5 * out_dim / batch_size * np.linalg.norm(out - Y_train)**2 for out in output_analytical]


plt.plot(losses_analytical, label='analytical')
plt.plot(losses_empirical, label='empirical')
plt.title('Loss trajectories for Analytical and Empirical Gradient Flow')
plt.legend()
plt.show()


##maybe transpose wrong?