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

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim,
        self.out_dim = out_dim,
        self.initial_scale = initial_scale,
        self.batch_size, _ = int(batch_size),
        self.learning_rate = learning_rate
        self.training_steps = training_steps


        self.tau = 1 / learning_rate

        #define weights for student teacher network - used for SIMULATION
        self.data_w1, self.data_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale * 2)
        self.train, _, _ = gs.datasets.StudentTeacher(batch_size, [self.data_w1, self.data_w2], [gs.datasets.Whiten()])

        (self.X, self.Y) = self.train(None)
        self.task = gs.tasks.FullBatchLearning(self.train)
        self.optimiser = gs.GradientDescent(self.learning_rate)
        self.loss = gs.MeanSquaredError()


        #define initial weights for training - used for SIMULATION AND ANALYTICAL
        self.init_w1, self.init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale)

        self.mlp = gs.Network([
            gs.Linear(hidden_dim, bias = False, weight_init=gs.init.FromFixedValue(self.init_w1)),
            gs.Linear(out_dim, bias=False, weight_init=gs.init.FromFixedValue(self.init_w2))
        ])

        print('batch size: ', self.batch_size)

        self.sigma_xy_target = (1 / self.batch_size) * self.Y @ self.X.T

        self.U_, self.S_, self.Vt_ = np.linalg.svd(self.sigma_xy_target)
        self.V_ = self.Vt_.T




        #for now zero balanced weights
        self.c = 0

        # have to define F, Q0,O
        self.q0 = np.vstack([self.init_w1.T,
                             self.init_w2])

        self.qqt0 = self.q0 @ self.q0.T

        self.F = np.vstack([
            np.hstack([self.c/2 * np.eye(self.sigma_xy_target.shape[1]), self.sigma_xy_target.T]),
            np.hstack([self.sigma_xy_target, self.c/2 * np.eye(self.sigma_xy_target.shape[0])])
        ])

        self.F_inv = np.linalg.inv(self.F_inv)

        self.O = 1/np.sqrt(2) * np.vstack([np.hstack([self.Vt_.T, self.Vt_.T]), np.hstack([self.U_, -self.U_])])

        self.lmda = np.vstack([
            np.hstack([self.S_, np.zeros(self.S_.shape)]),
            np.hstack([np.zeros(self.S_.shape), self.S_])
        ])





        #Defining all variables required for the analytical solution

