import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import torch 




#TODO: take in another parameter lambda f
class QQT:
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
        center = np.linalg.inv(center_left + center_center + center_right)

        qqt = z @ center @ z.T
        if self.weights_only:
            qqt = qqt[self.n_i:, :self.n_i]

        self.t += 1

        return qqt


class QQTDiagonal:
    def __init__(self, w1, w2, x_target, y_target):
        self.batch_size = x_target.shape[1]

        n_i = x_target.shape[0]
        n_o = y_target.shape[0]
        n_e = np.abs(n_o - n_i)

        whitened = np.all(np.round(1. / self.batch_size * x_target @ x_target.T, 3) == np.identity(n_i))
        identity = False
        if x_target.shape[0] == x_target.shape[1]:
            identity = np.all(x_target == np.identity(n_i))
        assert whitened or identity, f"X not whitened"

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

        self.b = u.T @ u_ + v.T @ v_
        self.c = u.T @ u_ - v.T @ v_

        self.t = 0


    def forward(self, learning_rate):
        tau = 1. / learning_rate
        tt = self.t / tau

        u, s, v = self.u, self.s, self.v
        u_, s_, v_ = self.u_, self.s_, self.v_

        c = np.diag(self.c)
        b = np.diag(self.b)
        s_ = np.diag(s_)
        s = np.diag(s)
        e2m = np.exp(-2. * s_ * tt)
        e4m = np.exp(-4. * s_ * tt)

        numerator = s * b ** 2 * s_ - s * c ** 2 * s_ * e4m
        denominator = 4. * s_ * e2m + s * b ** 2 * (1. - e2m) + s * c ** 2 * (e2m - e4m) + 1e-12

        self.t += 1.
        return u_ @ np.diag(numerator / denominator) @ v_.T


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

in_dim = 5
hidden_dim = 10
out_dim = 2

tau = 0.001
batch_size = 10
epochs = 1

init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, .35)
init_w1 = torch.tensor(init_w1)
init_w2 = torch.tensor(init_w2)

X_train, Y_train = get_random_regression_task(batch_size, in_dim, out_dim)

analytical = QQT(init_w1, init_w2, X_train.T, Y_train.T, True)
analytical = np.asarray([analytical.forward(tau) for _ in range(epochs)])