import gooseberry as gs

import matplotlib.pyplot as plt
import numpy as np

from tools import BlindColours, zero_balanced_weights


class QQTTask:

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 initial_scale,
                 batch_size,
                 learning_rate,
                 training_steps):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.initial_scale = initial_scale

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_steps = training_steps

        self.tau = 1 / learning_rate

        self.init_w1, self.init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale)
        self.train, _, _ = gs.datasets.StudentTeacher(batch_size, [self.init_w1, self.init_w2], [gs.datasets.Whiten()])

        # TODO: don't really understand this part
        (self.X, self.Y) = self.train(None)
        # self.train = (self.X, self.Y)
        self.c = 0

        self.q0 = np.vstack([self.init_w1.T,
                             self.init_w2])

        self.qqt0 = self.q0 @ self.q0.T

        self.sigma_xy = self.Y.T @ self.X

        self.F = np.vstack([
            np.hstack([self.c / 2 * np.eye(self.sigma_xy.shape[1]), self.sigma_xy.T]),
            np.hstack([self.sigma_xy, self.c / 2 * np.eye(self.sigma_xy.shape[0])])
        ])

        self.required_shape = (self.init_w2 @ self.init_w1).shape

        self.U_, self.S_, self.Vt_ = np.linalg.svd(self.sigma_xy)

        self.s = self.S_ + np.eye(self.S_.shape[0])

        self.lmda = np.vstack([
            np.hstack([self.s, np.zeros(self.s.shape)]),
            np.hstack([np.zeros(self.s.shape), self.s])
        ])

        self.lmda_inv = np.linalg.inv(self.lmda)

        self.e_f = np.exp(1 / self.tau * self.F)

        self.evals, self.evecs = np.linalg.eig(self.e_f)
        self.O = self.evecs
        self.e_lmda = np.diag(self.evals)

        self.U, self.A0, self.Vt = np.linalg.svd(self.init_w2 @ self.init_w1)
        # temp = np.zeros(self.A0.shape)

        self.A0 = np.diag(self.A0)
        self.V = self.Vt.T
        self.V_ = self.Vt_.T
        self.B = self.U.T @ self.U_ + self.V.T @ self.V_
        self.C = self.U.T @ self.U_ - self.V.T @ self.V_
        self.R = np.identity(self.A0.shape[0])

    def getw2w1(self, qqt):
        return qqt[-self.required_shape[0]:, :self.required_shape[1]]
