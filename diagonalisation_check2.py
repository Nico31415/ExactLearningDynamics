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



S = np.array([3, 5])

lmda = 1

theoretical_diag = np.vstack([
    np.hstack([np.diag(S), -lmda/2 * np.eye(S.shape[0])]),
    np.hstack([-lmda/2 * np.eye(S.shape[0]), -np.diag(S)])
])

# factor = np.sqrt(2 * np.sqrt(S**2 + 1) * (np.sqrt(S**2 + 1) + S))

factor = np.sqrt((8 * S**2 + 2 * lmda**2 + 4 * S * np.sqrt(lmda**2 + 4*S**2))/lmda**2)

off_diag = (2*S + np.sqrt(lmda**2 + 4*S**2)) / lmda

P = np.vstack([
    np.hstack([np.diag(1/factor)  ,            -np.diag(off_diag/factor)]),
    np.hstack([np.diag(off_diag/factor), np.diag(1/factor)])
])


expected_result = np.diag(np.concatenate((-np.sqrt(lmda**2/4 + S**2), np.sqrt(lmda**2/4 + S**2))))


print('P')