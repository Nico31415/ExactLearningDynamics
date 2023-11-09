
import numpy as np 
import gooseberry as gs

np.random.seed(1)


def whiten(X):
    u, _, vt = np.linalg.svd(X)
    return u @ np.eye(X.size) @ vt


def qqt_dynamics(training_data, in_dim, hidden_dim, w1, w2, out_dim, lmda, batch_size = 25):

    data = gs.datasets.RandomRegression(False, True)
 
    task = gs.tasks.FullBatchLearning(data)
    
    X = whiten(np.random.normal(0., 1., (in_dim, batch_size)))
    Y = target_computation.forward(X)

    sigma_xy = (1. / batch_size) * Y @ X.T 

    U, S, VT = np.linalg.svd(sigma_xy)

    print("U", U)
    print("S", S)
    print("VT", VT)
    ​
    U_ = U[:, 2:]
    U = U[:, :2]
    S = np.diag(S)
    V = VT.T
    ​
    print(U@S@V.T)
    print(sigma_xy)

    target_computation = LinearNetwork(in_dim, hidden_dim, out_dim, 1., 1.)
    X = whiten(np.random.normal(0., 1., (in_dim, batch_size)))
    Y = target_computation.forward(X)
    ​
    sigma_xy = (1. / batch_size) * Y @ X.T
    ​
    # Decompose sigma XY
    U, S, VT = np.linalg.svd(sigma_xy)
    ​
    print("U", U)
    print("S", S)
    print("VT", VT)
    ​
    U_ = U[:, 2:]
    U = U[:, :2]
    S = np.diag(S)
    V = VT.T
    ​
    print(U@S@V.T)
    print(sigma_xy)
    
    F = np.vstack([
        np.hstack([lmda / 2 * np.eye((sigma_xy.shape[1], sigma_xy.shape[1])), sigma_xy.T]),
        np.hstack([sigma_xy, lmda / 2 * np.eye((sigma_xy.shape[0], sigma_xy.shape[0]))])
    ])
    ​
    print(F)
    ​
    OD = 1. / np.sqrt(2) * np.vstack([
        np.hstack([V, V]),
        np.hstack([U, -U])
    ])
    ​
    OE = 1. / np.sqrt(2) * np.vstack([np.zeros((in_dim, U_.shape[1])), U_])
    ​
    LD = np.vstack([
        np.hstack([S, np.zeros_like(S)]),
        np.hstack([np.zeros_like(S), -S]),
    ])
    ​
    LE = np.zeros((U_.shape[1], U_.shape[1]))