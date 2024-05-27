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


in_dim = 2
hidden_dim = 2
out_dim = 2

batch_size = 10


X, Y = get_random_regression_task(batch_size, in_dim, out_dim)


sigma_yx = 1/batch_size * Y @ X.T 


lmda = 0.5

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

S_inv = 1/S

theoretical_diag = np.vstack([
    np.hstack([np.diag(S), -lmda/2 * np.eye(S.shape[0])]),
    np.hstack([-lmda/2 * np.eye(S.shape[0]), -np.diag(S)])
])

factor = np.diag(1/(np.sqrt((8 * S**2 + 2 * lmda**2 + 4 * S * np.sqrt(lmda**2 + 4*S**2))/lmda**2)))

off_diag = np.diag((2*S + np.sqrt(lmda**2 + 4*S**2)) / lmda)

P = np.vstack([
    np.hstack([factor  ,  -off_diag@factor]),
    np.hstack([off_diag @ factor, factor])
])

O2 = np.vstack([
    np.hstack([np.eye(S.shape[0]), np.zeros((S.shape[0], S.shape[0]))]),
    np.hstack([-lmda/2 * np.diag(1/S), np.eye(S.shape[0])])
])

# O_final = 1/np.sqrt(2) * np.vstack([
#     np.hstack([factor @ (np.eye(S.shape[0]) + off_diag) @ Vt, factor @ (np.eye(S.shape[0]) - off_diag) @ U.T ]),
#     np.hstack([factor @ (np.eye(S.shape[0]) - off_diag) @ Vt, -factor @ (np.eye(S.shape[0]) + off_diag) @ U.T])
# ])

# O_final = 1/np.sqrt(2) * np.vstack([
#     np.hstack([factor @ (np.eye(S.shape[0]) - off_diag) @ Vt, -factor @ (np.eye(S.shape[0]) + off_diag) @ U.T ]),
#     np.hstack([factor @ (np.eye(S.shape[0]) + off_diag) @ Vt, factor @ (np.eye(S.shape[0]) - off_diag) @ U.T])
# ])

# O_final = 1/np.sqrt(2) * np.vstack([
#     np.hstack([Vt.T @ (np.eye(S.shape[0]) - off_diag) @ factor, Vt.T @ (np.eye(S.shape[0]) + off_diag) @ factor]),
#     np.hstack([-U @ (np.eye(S.shape[0]) + off_diag) @ factor, U @ (np.eye(S.shape[0]) - off_diag) @ factor])
# ])




evals = np.sqrt(lmda**2/4 + S**2)

D = np.diag(np.concatenate((evals, -evals)))

# print(O2.T @ theoretical_diag @ O2)
# sns.heatmap(O2.T @ theoretical_diag @ O2)
# plt.show()


A = np.diag(np.sqrt(1/(1 + (lmda / (2 * S - np.sqrt(lmda**2 + 4*S**2)))**2)))

X = np.diag(lmda / (2 * S - np.sqrt(lmda**2 + 4 * S ** 2)))

I = np.eye(S.shape[0])

P = np.vstack([
    np.hstack([X @ A, A]),
    np.hstack([A, -X@A])
])

O_final = 1/np.sqrt(2) * np.vstack([
    np.hstack([Vt.T @ (X + I) @ A, Vt.T @ (X - I) @ A]),
    np.hstack([U @ (X - I) @ A, -U @ (X + I) @ A])
])

# O_final = 1/np.sqrt(2) * np.vstack([
#     np.hstack([Vt.T @ (I - X) @ A, Vt.T @ (I + X) @ A]),
#     np.hstack([U @ (I + X) @ A, -U @ (I - X) @ A])
# ])


# lmda = 100
# F = get_F(lmda)

O3 = 1/np.sqrt(2) * np.vstack([
    np.hstack([Vt.T @ (np.eye(S.shape[0]) - lmda / 2 * np.diag(1/S)), Vt.T]),
    np.hstack([U @ (np.eye(S.shape[0]) + lmda / 2 * np.diag(1/S)), -U])
    ])


O3 = 1/np.sqrt(2) * np.vstack([
    np.hstack([(np.eye(S.shape[0]) - lmda / 2 * np.diag(1/S)) @ Vt, (np.eye(S.shape[0]) + lmda / 2 * np.diag(1/S)) @ U.T]),
    np.hstack([Vt, -U.T])
    ])

# P = np.vstack([
#     np.hstack([np.diag(lmda/(2*S + np.sqrt(lmda**2+4*S**2))), np.eye(S.shape[0])]),
#     np.hstack([np.diag(lmda/(2*S - np.sqrt(lmda**2+4*S**2))), np.eye(S.shape[0])])
# ])

P = np.vstack([
    np.hstack([np.diag(lmda/(2*S - np.sqrt(lmda**2+4*S**2))), np.eye(S.shape[0])]),
    np.hstack([np.diag(lmda/(2*S - np.sqrt(lmda**2+4*S**2))), np.eye(S.shape[0])])
])

diagonal = np.vstack([
    np.hstack([np.diag(S) + lmda**2 /4 * np.diag(1/S), np.zeros((S.shape[0], S.shape[0]))]),
    np.hstack([np.zeros((S.shape[0], S.shape[0])), -np.diag(S)])
])

diff = np.linalg.norm(O3.T @ F @ O3 - diagonal)

print(diff)
print('hello')