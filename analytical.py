import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

class QQT:
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

        if self.t==10:
            print('QQT changed version:')
            print('V_: ', self.V_)
            print('U_: ', self.U_)
            print('e_st_inv: ', e_st_inv)
            print('C: ', self.C)
            print('B_inv: ', B_inv)
            print('V_hat: ', self.V_hat)
            print('V:  ', self.V)
            print('U_hat: ', self.U_hat)
            print('U:  ', self.U)

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
        
        if self.t==1:
            print('New Version center_center variables: ')
            print('i: ', i)
            print('e_2st_inv: ', e_2st_inv)
            print('S_inv: ', S_inv)
            print('e_st_inv: ', e_st_inv)
            print('B_inv: ', B_inv)
            print('C: ', self.C)

        center = np.linalg.inv(center_left + center_center + center_right)

        qqt = Z @ center @ Z.T 
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 


import torch

def zero_balanced_weights(in_dim, hidden_dim, out_dim, sigma):
    r, _, _ = np.linalg.svd(np.random.normal(0., 1., (hidden_dim, hidden_dim)))

    w1 = np.random.normal(0., sigma, (hidden_dim, in_dim))
    w2 = np.random.normal(0., sigma, (out_dim, hidden_dim))
    u, s, vt = np.linalg.svd(w2 @ w1, False)

    s = np.diag(np.sqrt(s) * 1.15)

    smaller_dim = in_dim if in_dim < out_dim else out_dim

    s0 = np.vstack([s, np.zeros((hidden_dim - smaller_dim, smaller_dim))])
    w1 = r @ s0 @ vt

    s0 = np.hstack([s, np.zeros((smaller_dim, hidden_dim - smaller_dim))])
    w2 = u @ s0 @ r.T

    return w1, w2

def whiten(X):

    scaler = StandardScaler()
    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    X_whitened = torch.tensor(X_pca / np.sqrt(pca.explained_variance_), dtype=torch.float32)

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = torch.randn(batch_size, in_dim)
    Y = torch.randn(batch_size, out_dim)
    X_whitened = whiten(X)
    if batch_size > 1:
        X_whitened = 1/np.sqrt(batch_size - 1) * X_whitened

    return X_whitened, Y


class QQT_original:
    def __init__(self, w1, w2, x_target, y_target, weights_only=False):
        self.weights_only = weights_only
        self.batch_size = x_target.shape[1]

        self.n_i = n_i = x_target.shape[0]
        self.n_o = n_o = y_target.shape[0]
        n_e = np.abs(n_o - n_i)

        whitened = np.all(np.round(1. / self.batch_size * x_target @ x_target.T, 3) == np.identity(n_i))
        identity = False
        # if x_target.shape[0] == x_target.shape[1]:
        #     identity = np.all(x_target == np.identity(n_i))
        # assert whitened or identity, f"X not whitened"

        sigma_xy_target = (1. / self.batch_size) * y_target @ x_target.T

        u_, s_, vt_ = np.linalg.svd(sigma_xy_target)
        v_ = vt_.T

        if n_i < n_o:
            u_hat = u_[:, n_i:]
            v_hat = np.zeros((n_i, n_e))
            u_ = u_[:, :n_i]
        elif n_i > n_o:
            u_hat = np.zeros((n_o, n_e))
            v_hat = v_[:, n_o:]
            v_ = v_[:, :n_o]
        else:
            u_hat = v_hat = None

        self.u_hat = u_hat
        self.v_hat = v_hat
        self.u_, self.s_, self.v_ = u_, np.diag(s_), v_


        u, s, vt = np.linalg.svd(w2 @ w1, False)
        v = vt.T
        self.u, self.s, self.v = u, np.diag(s), v

        b = u.T @ u_ + v.T @ v_
        assert np.abs(np.linalg.det(b)) > 1e-10, f"B is not invertible det(B) = {np.linalg.det(b)}"
        self.b_inv = np.linalg.inv(b)
        self.c = u.T @ u_ - v.T @ v_

        self.i = np.identity(n_i) if n_i < n_o else np.identity(n_o)

        self.t = 0

    def forward(self, learning_rate):
        tau = 1. / learning_rate
        tt = self.t / tau

        i = self.i
        u, s, v = self.u, self.s, self.v
        u_, s_, v_ = self.u_, self.s_, self.v_

        c = self.c
        b_inv = self.b_inv

        v_hat, u_hat = self.v_hat, self.u_hat

        em = np.diag(np.exp(-1. * np.diag(s_) * tt))
        e2m = np.diag(np.exp(-2. * np.diag(s_) * tt))

        s_inv = np.diag(1. / np.diag(s))
        s__inv = np.diag(1. / np.diag(s_))

        if self.t ==10:
            print('QQT original version:')
            print('V_: ', v_)
            print('U_: ', u_)
            print('e_st_inv: ', em)
            print('C: ', c)
            print('B_inv: ', b_inv)
            print('V_hat: ', v_hat)
            print('V:  ', v)
            print('U_hat: ', u_hat)
            print('U:  ', u)

        if u_hat is None and v_hat is None:
            z = np.vstack([
                v_ @ (i - em @ c.T @ b_inv.T @ em),
                u_ @ (i + em @ c.T @ b_inv.T @ em)
            ])
            center_right = 0.
        else:
            z = np.vstack([
                v_ @ (i - em @ c.T @ b_inv.T @ em) + 2. * v_hat @ v_hat.T @ v @ b_inv.T @ em,
                u_ @ (i + em @ c.T @ b_inv.T @ em) + 2. * u_hat @ u_hat.T @ u @ b_inv.T @ em
            ])
            center_right = 4 * tt * em @ b_inv @ (v.T @ v_hat @ v_hat.T @ v + u.T @ u_hat @ u_hat.T @ u) @ b_inv.T @ em

        center_left = 4. * em @ b_inv @ s_inv @ b_inv.T @ em
        center_center = (i - e2m) @ s__inv - em @ b_inv @ c @ (e2m - i) @ s__inv @ c.T @ b_inv.T @ em

        if self.t==1:
            print('Original Version center_center variables: ')
            print('i: ', i)
            print('e_2st_inv: ', e2m)
            print('S_inv: ', s__inv)
            print('e_st_inv: ', em)
            print('B_inv: ', b_inv)
            print('C: ', c)

        center = np.linalg.inv(center_left + center_center + center_right)

        qqt = z @ center @ z.T
        if self.weights_only:
            qqt = qqt[self.n_i:, :self.n_i]

        self.t += 1

        return qqt

in_dim = 5
hidden_dim = 10
out_dim = 2

tau = 0.001
batch_size = 10
epochs = 100

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
init_w1 = torch.tensor(init_w1)
init_w2 = torch.tensor(init_w2)

X_train, Y_train = get_random_regression_task(batch_size, in_dim, out_dim)

analytical2 = QQT_original(init_w1, init_w2, X_train.T, Y_train.T, True)
analytical2 = np.asarray([analytical2.forward(tau) for _ in range(epochs)])

analytical = QQT(init_w1, init_w2, X_train, Y_train, True)
analytical = np.asarray([analytical.forward(tau) for _ in range(epochs)])

diffs = [np.linalg.norm(a1 - a2) for (a1, a2) in zip(analytical2, analytical)]
print(diffs[-1])



