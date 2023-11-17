import gooseberry as gs

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from dynamics import QQT
from dynamics import QQT_balanced
from tools import BlindColours, zero_balanced_weights


def plot_matrix_evolution(matrix_list, n_components):
    """
    Plot the evolution of the first n_components of a matrix through time in logarithmic scale.

    Parameters:
    - matrix_list: List of matrices representing the evolution of the matrix through time.
    - n_components: Number of components to plot.

    Returns:
    None (displays the plot).
    """
    time_steps = len(matrix_list)

    # Extract the first n_components from each matrix
    components_evolution = np.array([matrix[:n_components, :n_components] for matrix in matrix_list])

    # Create a figure with logarithmic x-axis
    plt.figure(figsize=(10, 6))

    # Plot the evolution of each component in logarithmic scale
    for i in range(n_components):
        plt.semilogx(range(1, time_steps + 1), components_evolution[:, i, i], label=f'Component {i + 1}')

    # Set labels and title
    plt.xlabel('Time Step (log scale)')
    plt.ylabel('Component Value')
    plt.title(f'Evolution of the First {n_components} Components (log scale)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def train_network(train, learning_rate, hidden_dim, out_dim, init_w1, init_w2, training_steps):
    # Generate the computational solution
    task = gs.tasks.FullBatchLearning(train)
    optimiser = gs.GradientDescent(learning_rate)
    loss = gs.MeanSquaredError()

    mlp = gs.Network([
        gs.Linear(hidden_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w1)),
        gs.Linear(out_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w2))
    ])

    trainer = gs.Trainer(task, mlp, loss, optimiser)
    state, params = gs.assemble(1)

    losses = []
    ws = [params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]]

    for training_step in range(training_steps):
        state, params, current_loss = trainer(state, params)
        losses.append(current_loss)
        ws.append(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])

    return losses, ws


def check_analytical_solution(solution):
    """
    Checks analytical solution from paper against computational result.
    Input: number of equation on the paper

    3: 1/tau (FQQt + QQtF - (QQt)^2)

    4: exponent stuff, doesnt converge well
    """
    in_dim = 5
    hidden_dim = 10
    out_dim = 5
    initial_scale = 0.35

    batch_size = 10
    learning_rate = 0.1
    training_steps = 400

    ##TODO: not sure what to call tau
    tau = 1 / learning_rate

    init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, initial_scale)
    train, _, _ = gs.datasets.StudentTeacher(batch_size, [init_w1, init_w2], [gs.datasets.Whiten()])

    (X, Y) = train(None)

    plot_items_n = 4
    blind_colours = BlindColours().get_colours()
    c = 0

    Q0 = np.vstack([init_w1.T, init_w2])

    QQt0 = Q0 @ Q0.T

    sigma_xy = Y.T @ X

    F = np.vstack([
        np.hstack([c / 2 * np.eye(sigma_xy.shape[1]), sigma_xy.T]),
        np.hstack([sigma_xy, c / 2 * np.eye(sigma_xy.shape[0])])
    ])

    required_shape = (init_w2 @ init_w1).shape
    QQts = [QQt0]
    w2w1s = [[QQt0[-required_shape[0]:, :required_shape[1]]]]

    U_, S_, Vt_ = np.linalg.svd(sigma_xy)

    s = S_ + np.eye(S_.shape[0])

    O = ...
    lmda = np.vstack([
        np.hstack([s, np.zeros(s.shape)]),
        np.hstack([np.zeros(s.shape), s])
    ])

    lmda_inv = np.linalg.inv(lmda)

    e_f = np.exp(1 / tau * F)

    evals, evecs = np.linalg.eig(e_f)
    O = evecs
    e_lmda = np.diag(evals)

    U, A0, Vt = np.linalg.svd(init_w2 @ init_w1)
    V = Vt.T
    V_ = Vt_.T
    B = U.T @ U_ + V.T @ V_
    C = U.T @ U_ - V.T @ V_
    R = np.identity(A0.shape[0])

    if solution == '3':
        for i in range(1, training_steps):
            curr = QQts[-1]

            derivative = F @ curr + curr @ F - curr @ curr.T
            next = curr + learning_rate * derivative
            QQts.append(next)

            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))


    elif solution == '4':
        for i in range(1, training_steps):
            tau = 1
            t = i * learning_rate
            e_ft = np.exp(t / tau * F)

            out = e_ft @ Q0
            centre_centre = e_ft @ np.linalg.inv(F) @ e_ft - np.linalg.inv(F)
            # print(np.eye(F.shape[0]) + 1/2 * Q0.T @ centre_centre @ Q0)
            try:
                centre = np.linalg.inv(np.eye(F.shape[0]) + 1 / 2 * Q0.T @ centre_centre @ Q0)
            except:
                print(i)
                return
            next = out @ centre @ out.T

            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))


    elif solution == '10':
        for i in range(1, training_steps):
            t = i * learning_rate
            print(e_lmda)
            print(t)
            e_lmdat = e_lmda ** t

            left = O @ e_lmdat @ O.T @ Q0
            right = left.T

            centre = np.linalg.inv(np.eye(lmda_inv.shape[0]) + 1 / 2 * Q0.T @ (
                        O @ e_lmdat @ O.T @ O @ lmda_inv @ O.T @ e_lmdat @ O.T - O @ lmda_inv @ O.T))

            next = left @ centre @ right
            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))


    elif solution == '12':
        for i in range(1, training_steps):
            t = i * learning_rate
            e_lmdat = e_lmda ** t

            left = O @ e_lmdat @ O.T @ Q0
            right = left.T

            centre = np.linalg.inv(np.eye(lmda_inv.shape[0]) + 1 / 2 * Q0.T @ (
                        O @ e_lmdat @ lmda_inv @ e_lmdat @ O.T - O @ lmda_inv @ O.T) @ Q0)

            next = left @ centre @ right
            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))

    elif solution == '13':
        for i in range(1, training_steps):
            t = i * learning_rate
            e_lmdat = e_lmda ** t
            e_lmdat = e_lmda ** (2 * t)

            left = O @ e_lmdat @ O.T @ Q0
            right = left.T

            centre = np.linalg.inv(
                np.eye(lmda_inv.shape) + 1 / 2 * Q0.T @ O @ (e_2lmdat @ lmda_inv - lmda_inv) @ O.T @ Q0)

            next = left @ centre @ right
            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))
        return

    elif solution == '14':
        for i in range(1, training_steps):
            t = i * learning_rate
            e_lmdat = np.exp(e_lmda, t)
            e_2lmdat = np.exp(e_lmda, 2 * t)

            left = O @ e_lmdat @ O.T @ Q0
            right = left.T

            centre = np.linalg.inv(
                np.eye(...) + 1 / 2 * Q0.T @ O @ (e_2lmdat - np.eye(e_2lmdat.shape[0]) @ lmda_inv @ O.T @ Q0))

            next = left @ centre @ right
            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))



    elif solution == '30':

        for i in range(1, training_steps):
            e_st = np.exp(s, t / tau)
            e_st_inv = np.linalg.inv(e_st)
            root_A0 = A0 ** 0.5

            left = 1 / 2 * np.vstack([
                V_ @ (e_st @ B.T - e_st_inv @ C.T) @ root_A0 @ R.T,
                U_ @ (e_st @ B.T + e_st_inv @ C.T) @ root_A0 @ R.T,
            ])

            right = left.T

            centre_centre = ...

            centre = np.linalg.inv(np.eye(...) + 1 / 4 @ R @ root_A0 @ centre_centre @ root_A0 @ R.T)
    ##TODO: i think this one is the hardest to implement
    ##TODO: A(0), RT

    elif solution == '37':
        for i in range(1, training_steps):
            t = i * learning_rate
            e_lmdat = np.exp(e_lmda, t)
            e_2lmdat = np.exp(e_lmda, 2 * t)
            e_st_inv = np.linalg.inv(e_st)
            B_inv = np.lianlg.inv(B)
            A0 = ...

            left = O @ e_lmdat @ O.T @ Q0
            right = left.T

            left = np.vstack([
                V_ @ (np.eye(...) - e_st_inv @ C.T @ np.linalg.inv(B).T @ e_st_inv),
                U_ @ (np.eye(...) - e_st_inv @ C.T @ np.linalg.inv(B).T @ e_st_inv)
            ])
            right = left.T
            centre_left = 4 * e_st_inv @ B_inv @ A0 @ B_inv.T @ e_st_inv
            centre_centre = (np.eye(s.shape[0]) - e_st_inv ** 2) @ np.linalg.inv(s)
            centre_right = - e_st_inv @ B_inv @ C @ (e_st_inv ** 2 - np.eye(s)) @ np.linalg.inv_s @ C.T @ B_inv.T @ e_st_inv

            centre = centre_left + centre_centre + centre_right

            next = left @ centre @ right
            QQts.append(next)
            w2w1s.append(np.array([next[-required_shape[0]:, :required_shape[1]]]))

# Now, generate the computational solution
losses, ws = train_network(train, learning_rate, hidden_dim, out_dim, init_w1, init_w2, training_steps)

analytical_output = (np.asarray(ws)[:, 0, :, :] @ X[:plot_items_n].T)
simulation_output = (np.asarray(w2w1s)[:, 0, :, :] @ X[:plot_items_n].T)

print('analytical shape: ', analytical_output.shape)
print('simulation shape: ', simulation_output.shape)

"""
Plot trajectories of the representations of both
"""
rng = np.linspace(0.57, 1., 10)
# TODO: plot analytical output


plt.figure()
# fig.set_xscale('log')

plt.xlabel('Training Steps')
plt.title('Simulation Results')
# for color, output in zip(blind_colours, simulation_output[1].T):
#     for val in output:
#         plt.plot(rng, [val]*10, c=color, lw=2.5, clip_on=False, zorder=1)
#         plt.plot(rng, [val]*10, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), clip_on=False, zorder=2)
plot_matrix_evolution(simulation_output, 4)

# TODO: plot simulation output
for n, (color) in enumerate(blind_colours[:plot_items_n]):
    plt.plot(-5, -5, c=color, lw=2.5, label=f"Item {n + 1}")
plt.plot(-5, -5, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label="Analytical")

"""
Plot difference between analytical and simulation
"""
diff = np.array([m1 - m2 for m1, m2 in zip(analytical_output, simulation_output)])

plt.figure()
plt.title('difference between simulation and analytical')
for color, output in zip(blind_colours, diff[1].T):
    for val in output:
        plt.plot(rng, [val] * 10, c=color, lw=2.5, clip_on=False, zorder=1)
        plt.plot(rng, [val] * 10, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), clip_on=False, zorder=2)

plt.show()

# print('analytical: ', analytical_output)
# print('simulation: ', simulation_output)
# print('diff: ', diff)
return

equation_number = input('what equation should we check? Input a number: ')
check_analytical_solution(equation_number)