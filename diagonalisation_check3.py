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

from empiricalTest import QQT_new
from empiricalTest import LinearNetwork
from empiricalTest import get_random_regression_task

in_dim = 5
hidden_dim = 5
out_dim = 5

batch_size = 10

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)


sigma_yx = 1/batch_size * Y @ X.T 


lmda = 100

def get_F(lmda):
    F = np.vstack([
        np.hstack([-lmda / 2 * np.eye(sigma_yx.shape[1]), sigma_yx.T]),
        np.hstack([sigma_yx, lmda / 2 * np.eye(sigma_yx.shape[0])])
    ]) 
    return F 

F = get_F(lmda)

U, S, Vt = np.linalg.svd(sigma_yx)


O1 = 1/np.sqrt(2) * np.vstack([
    np.hstack([Vt.T, Vt.T]),
    np.hstack([U, -U])
    ])


theoretical_diag = np.vstack([
    np.hstack([np.diag(S), -lmda/2 * np.eye(S.shape[0])]),
    np.hstack([-lmda/2 * np.eye(S.shape[0]), -np.diag(S)])
])

X = (np.sqrt(lmda**2 + 4*S**2) - 2* S)/lmda
A = np.diag(1 / (np.sqrt(1 + X**2)))

X = np.diag(X)

P = np.vstack([
    np.hstack([A, X @ A]),
    np.hstack([-X @ A, A])
])

O_final = 1/np.sqrt(2) * np.vstack([
    np.hstack([Vt.T @ (A - X@ A), Vt.T @ (A + X@A)]),
    np.hstack([U @ (A + X @ A), - U @ (A - X@ A)])
])

evals = np.sqrt(lmda**2/4 + S**2)

D = np.diag(np.concatenate((evals, -evals)))

print(P.T @ O1.T @ F @ O1 @ P)