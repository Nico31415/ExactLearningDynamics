import numpy as np 


class QQT:
    def __init__(self, w1, w2, x_target, y_target, weights_only = False):
        self.weights_only = weights_only
        self.batch_size = x_target.shape[1]

        self.n_i = n_i = x_target.shape[0]
        self.n_o = n_o = y_target.shape[0]
        n_e = np.abs(n_o - n_i)
        
        whitened = np.all(np.rouund(1. / self.batch_size * x_target @ x_target.T, 3) == np.identity(n_i))

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


        b = u.T @ u_ + v.T @ v_ 
        assert np.abs(np.linalg.det(b)) > 1e-10, f"B is not invertible det(B) = {np.linalg.det(b)}"
        self.b_inv = np.linalg.inv(b)
        self.c = u.T @ u_ - v.T @ v_ 

        self.i = np.identity(n_i) if n_i < n_o else np.identity(n_o)

        self.t = 0

    def forward(self, learning_rate):
        tau = 1. / learning_rate
        tt = self.t/tau 

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

            center_right = 4 * tt * em @ b_inv @ (v.T @ v_hat @v_hat.T @ v + u.T @ u_hat @u_hat.T @ u) @ b_inv.T @ em 


        center_left = 4. * em @ b_inv @ s_inv @ b_inv.T @ em 
        center_center = (i - e2m) @ s__inv - em @ b_inv @ c @ (e2m - i) @ s__inv @ c.T @ b_inv.T @ em 
        center = np.linalg.inv(center_left + center_center + center_right)

        qqt = z @ center @ z.T 
        if self.weights_only:
            qqt = qqt[self.n_i:, :self.n_i]
        
        self.t += 1

        return qqt