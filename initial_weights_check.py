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



in_dim = 5
hidden_dim = 5
out_dim = 5

lmda = 1

a = (np.sqrt(lmda) + 1)
b = np.sqrt(2*np.sqrt(lmda) + 1)

init_w2 = np.eye(hidden_dim)*a
init_w1 = np.eye(hidden_dim)*b


U, S, Vt = np.linalg.svd(init_w2 @ init_w1)

S_2 = np.sqrt((lmda + np.sqrt(lmda**2 + 4*S**2)) / 2)
S_1 = S / S_2 

print('hi')