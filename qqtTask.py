import gooseberry as gs

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power

from tools import BlindColours, zero_balanced_weights
from scipy.linalg import fractional_matrix_power


class QQTTask:

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 initial_scale,
                 batch_size,
                 learning_rate,
                 training_steps):

        self.plot_items_n = 4
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.initial_scale = initial_scale

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_steps = training_steps

        self.tau = 1 / learning_rate

        self.init_w1, self.init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale * 2)

        self.data_w1, self.data_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale * 2)
        self.train, _, _ = gs.datasets.StudentTeacher(batch_size, [self.data_w1, self.data_w2], [gs.datasets.Whiten()])

        self.blind_colours = BlindColours().get_colours()
        # TODO: don't really understand this part
        (self.X, self.Y) = self.train(None)

        self.X = self.X.T
        self.Y = self.Y.T

        whitened = np.all(np.round(1. / self.batch_size * self.X @ self.X.T, 2) == np.identity(in_dim))

        print('difference: ', np.round(1. / self.batch_size * self.X @ self.X.T, 4)  - np.identity(in_dim))
        print(self.X @ self.X.T)
        # assert whitened, f'X not whitened'
        # self.train = (self.X, self.Y)
        self.c = 0

        self.q0 = np.vstack([self.init_w1.T,
                             self.init_w2])

        self.qqt0 = self.q0 @ self.q0.T

        self.sigma_xy =  (1 / self.batch_size) * self.Y @ self.X.T

        self.F = np.vstack([
            np.hstack([self.c / 2 * np.eye(self.sigma_xy.shape[1]), self.sigma_xy.T]),
            np.hstack([self.sigma_xy, self.c / 2 * np.eye(self.sigma_xy.shape[0])])
        ])

        self.required_shape = (self.init_w2 @ self.init_w1).shape

        self.U_, self.S_, self.Vt_ = np.linalg.svd(self.sigma_xy)

        self.s = self.S_ + np.eye(self.S_.shape[0])
        self.s_inv = np.linalg.inv(self.s)

        self.lmda = np.vstack([
            np.hstack([self.s, np.zeros(self.s.shape)]),
            np.hstack([np.zeros(self.s.shape), self.s])
        ])

        self.lmda_inv = np.linalg.inv(self.lmda)

        self.task = gs.tasks.FullBatchLearning(self.train)
        self.optimiser = gs.GradientDescent(self.learning_rate)
        self.loss = gs.MeanSquaredError()

        self.e_f = np.exp(1 / self.tau * self.F)

        self.evals, self.evecs = np.linalg.eig(self.e_f)


        self.O = 1/np.sqrt(2) * np.vstack([np.hstack([self.Vt_.T, self.Vt_.T]), np.hstack([self.U_, -self.U_])])
        # self.O = self.evecs

        self.e_lmda = fractional_matrix_power(self.lmda, np.e)
        # self.e_lmda = np.diag(self.evals)

        self.U, self.A0, self.Vt = np.linalg.svd(self.init_w2 @ self.init_w1)
        # temp = np.zeros(self.A0.shape)

        self.A0 = np.diag(self.A0)
        self.rootA0 = fractional_matrix_power(self.A0, 0.5)


        self.V = self.Vt.T
        self.V_ = self.Vt_.T
        self.B = self.U.T @ self.U_ + self.V.T @ self.V_
        self.C = self.U.T @ self.U_ - self.V.T @ self.V_
        self.R = np.identity(self.A0.shape[0])

    def getw2w1(self, qqt):
        return qqt[-self.required_shape[0]:, :self.required_shape[1]]
