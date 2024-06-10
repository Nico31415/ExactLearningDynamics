import numpy as np 
from utils import get_random_regression_task, get_lambda_balanced
from linear_network import LinearNetwork
import matplotlib.pyplot as plt 
import seaborn as sns 


class Aligned_Dynamics:
    def __init__(self, init_w1, init_w2, X, Y):
        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]

        sigma_yx_tilde = 1/ X.shape[1] * Y @ X.T

        self.U, self.S, self.Vt = np.linalg.svd(sigma_yx_tilde)

        self.a0 = np.diag(self.U.T @ init_w2 @ init_w1 @ self.Vt.T)

        self.theta0 = np.arcsinh(2/self.lmda * np.diag(self.U.T @ init_w2 @ init_w1 @ self.Vt.T))

        self.K = np.sqrt(self.lmda**2 + 4 * self.S**2)

        term = np.sign(self.lmda) * 2 * self.a0 / (np.sqrt(self.lmda**2+4*self.a0**2) + np.abs(self.lmda))

        self.c0 = (self.K + self.lmda + 2*self.S*term) / (self.K - self.lmda - 2 * self.S * term)

        # self.c02 = (self.K + self.lmda + 2*self.S*np.tanh(self.theta0/2)) / (self.K - self.lmda - 2*self.S*np.tanh(self.theta0/2))
        
        self.t = 0
        
        
    def forward(self, learning_rate):
        time_step = self.t * learning_rate

        K_exp = np.exp(self.K * time_step * np.sign(self.lmda))

        numerator = self.K * (self.c0 * K_exp - 1) - self.lmda * (self.c0 * K_exp+1)
        denominator = 2 * self.S * (self.c0 * K_exp + 1)

        # numerator = self.K * (self.c02 * K_exp - 1) - self.lmda * (self.c02 * K_exp+1)
        # denominator = 2 * self.S * (self.c02 * K_exp + 1)

        # theta_t = 2 * np.arctanh(numerator / denominator)
        x = numerator / denominator 

        a_t = self.lmda * x / (1 - x**2)

        self.t += 1 
        # return theta_t

        return a_t 

in_dim = 2
hidden_dim = 2
out_dim = 2

batch_size = 5
learning_rate = 0.01
training_steps = 1000

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)
U, S, Vt = np.linalg.svd(1/batch_size*Y@X.T)

lmda = -1
init_w1_hat, init_w2_hat = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)

U_, _, _ = np.linalg.svd(init_w2_hat)
_, _, Vt_ = np.linalg.svd(init_w1_hat)

init_w2 = U @ U_.T @ init_w2_hat 
init_w1 = init_w1_hat @ Vt_.T @ Vt 

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)

as_list = [np.diag(U.T @ w2 @ w1 @ Vt.T) for (w2, w1) in zip(w2s, w1s)]

thetas_list = [np.arcsinh(2/lmda * a) for a in as_list]


aligned_dynamics = Aligned_Dynamics(init_w1, init_w2, X, Y)

analytical_dynamics = [aligned_dynamics.forward(learning_rate) for _ in range(training_steps)]


print('hi')

##THETAS WORK, NOW DO IT IN TERMS OF ALPHAS, 