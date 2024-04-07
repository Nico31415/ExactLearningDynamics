import gooseberry as gs

import matplotlib.pyplot as plt
import numpy as np

from tools import BlindColours, zero_balanced_weights
from scipy.linalg import fractional_matrix_power
from qqtTask import QQTTask


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


def train_network(train, learning_rate, in_dim, hidden_dim, out_dim, init_w1, init_w2, training_steps):


    task = gs.tasks.FullBatchLearning(train)


    optimiser = gs.GradientDescent(learning_rate)
    loss = gs.MeanSquaredError()

    # init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.5)

    mlp = gs.Network([
        gs.Linear(hidden_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w1)),
        gs.Linear(out_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w2))
    ])

    trainer = gs.Trainer(task, mlp, loss, optimiser)
    state, params = gs.assemble(1)
    losses = []
    ws = []

    ws = [params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]]

    for training_step in range(training_steps):
        state, params, loss = trainer(state, params)
        losses.append(loss)
        ws.append(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])

    return losses, ws


# generates the computational solution using gradient descent
# def train_network(train, learning_rate, in_dim, hidden_dim, out_dim, init_w1, init_w2, training_steps):
#     # Generate the computational solution
#     task = gs.tasks.FullBatchLearning(train)
#     optimiser = gs.GradientDescent(learning_rate)
#     loss = gs.MeanSquaredError()
#
#     mlp = gs.Network([
#         gs.Linear(hidden_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w1)),
#         gs.Linear(out_dim, bias=False, weight_init=gs.init.FromFixedValue(init_w2))
#     ])
#
#     trainer = gs.Trainer(task, mlp, loss, optimiser)
#     state, params = gs.assemble(1)
#
#     losses = []
#
#     ws = np.zeros((training_steps, in_dim, out_dim))
#     ws[0] = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]
#     # ws = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]
#
#     for training_step in range(training_steps - 1):
#         # print('step: ', training_step)
#         state, params, current_loss = trainer(state, params)
#         losses.append(current_loss)
#         # ws.append(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])
#         # print(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])
#         ws[training_step + 1] = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]
#         # np.append(ws, params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])
#
#     return losses, ws


def check_analytical_solution(solution, qqtTask):
    # takes in solution (string) number on paper "Follow Up Deep Linear Network"
    """
    Checks analytical solution from paper against computational result.
    Input: number of equation on the paper

    3: 1/tau (FQQt + QQtF - (QQt)^2)

    4: exponent stuff, doesn't converge well
    """

    QQts = [qqtTask.qqt0]
    w2w1s = [qqtTask.getw2w1(qqtTask.qqt0)]
    losses = []

    if solution == '3':
        for i in range(1, qqtTask.training_steps):
            curr = QQts[-1]

            derivative = qqtTask.F @ curr + curr @ qqtTask.F - curr @ curr.T
            next_val = curr + qqtTask.learning_rate * derivative
            QQts.append(next_val)

            w2w1s.append(qqtTask.getw2w1(next_val))

            curr_weight = w2w1s[-1]

            print('Dimensions Y: ', qqtTask.Y.shape)
            print('Dimensions curr weight: ', curr_weight.shape)
            print('Dimensions task X: ', qqtTask.X.shape)

            # fitted = curr_weight @ qqtTask.X
            #
            # print('Dimensions fitted Y: ', fitted.shape)
            loss = (1 / 2) * np.linalg.norm(qqtTask.Y - np.matmul(qqtTask.X, curr_weight)) ** 2
            losses.append(loss)

    elif solution == '4':
        for i in range(1, qqtTask.training_steps):
            t = i
            e_ft = np.exp((t / qqtTask.tau) * qqtTask.F)

            out = e_ft @ qqtTask.q0
            centre_centre_centre = (e_ft @ np.linalg.inv(qqtTask.F) @ e_ft - np.linalg.inv(qqtTask.F))
            centre_centre = (1 / 2 * qqtTask.q0.T @ centre_centre_centre @ qqtTask.q0)
            print('centre centre shape: ', centre_centre.shape)
            centre = np.linalg.inv(np.eye(centre_centre.shape[0]) + centre_centre)
            # try:
            #     centre = np.linalg.inv(np.eye(qqtTask.F.shape[0]) + 1 / 2 * qqtTask.q0.T @ centre_centre @ qqtTask.q0)
            # except:
            #     print(i)
            #     break
            next_val = out @ centre @ out.T

            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))
            curr_weight = w2w1s[-1]
            loss = (1 / 2) * np.linalg.norm(qqtTask.Y - np.matmul(qqtTask.X, curr_weight)) ** 2
            losses.append(loss)

    elif solution == '10':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            print(qqtTask.e_lmda)
            print(t)
            e_lmdat = qqtTask.e_lmda ** t

            left = qqtTask.O @ e_lmdat @ qqtTask.O.T @ qqtTask.q0
            right = left.T

            centre = np.linalg.inv(np.eye(qqtTask.lmda_inv.shape[0]) + 1 / 2 * qqtTask.q0.T @ (
                    qqtTask.O @ e_lmdat @ qqtTask.O.T @ qqtTask.O @ qqtTask.lmda_inv @ qqtTask.O.T @ e_lmdat
                    @ qqtTask.O.T - qqtTask.O @ qqtTask.lmda_inv @ qqtTask.O.T))
            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))
            loss = (1 / 2) * np.linalg.norm(qqtTask.Y - np.matmul(qqtTask.X, curr_weight)) ** 2
            losses.append(loss)

    elif solution == '12':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            e_lmdat = qqtTask.e_lmda ** t

            left = qqtTask.O @ e_lmdat @ qqtTask.O.T @ qqtTask.q0
            right = left.T

            centre = np.linalg.inv(np.eye(qqtTask.lmda_inv.shape[0]) + 1 / 2 * qqtTask.q0.T @ (
                    qqtTask.O @ e_lmdat @ qqtTask.lmda_inv @ e_lmdat @ qqtTask.O.T
                    - qqtTask.O @ qqtTask.lmda_inv @ qqtTask.O.T) @ qqtTask.q0)

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))

    elif solution == '13':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            print('t: ', t)
            print('e_lmda: ', qqtTask.e_lmda)
            e_lmdat = fractional_matrix_power(qqtTask.e_lmda, t)
            e_2lmdat = fractional_matrix_power(qqtTask.e_lmda, 2 * t)

            left = qqtTask.O @ e_lmdat @ qqtTask.O.T @ qqtTask.q0
            right = left.T

            centre = np.linalg.inv(
                np.eye(qqtTask.lmda_inv.shape[0]) +
                1 / 2 * (qqtTask.q0.T @ qqtTask.O @
                         (e_2lmdat @ qqtTask.lmda_inv - qqtTask.lmda_inv) @ qqtTask.O.T @ qqtTask.q0))

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))
        return


    #student teacher guarantees solution
    #if you are full rank you are always guaranteed

    elif solution == '14':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            e_lmdat = fractional_matrix_power(qqtTask.e_lmda, t)
            e_2lmdat = fractional_matrix_power(qqtTask.e_lmda, 2 * t)

            left = qqtTask.O @ e_lmdat @ qqtTask.O.T @ qqtTask.q0
            right = left.T

            centre = np.linalg.inv(

                # TODO: i think the dimensions are correct here, might have to double check
                np.eye(qqtTask.hidden_dim) +
                1 / 2 * (qqtTask.q0.T @ qqtTask.O @
                         (e_2lmdat - np.eye(e_2lmdat.shape[0]) @ qqtTask.lmda_inv @ qqtTask.O.T @ qqtTask.q0))
            )

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))

    elif solution == '30':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            e_st = fractional_matrix_power(qqtTask.s, t / qqtTask.tau)
            e_st_inv = np.linalg.inv(e_st)
            print('A0 shape: ', qqtTask.A0.shape)
            print('A0: ', qqtTask.A0)

            left = 1 / 2 * np.vstack([
                qqtTask.V_ @ (e_st @ qqtTask.B.T - e_st_inv @ qqtTask.C.T) @ qqtTask.rootA0 @ qqtTask.R.T,
                qqtTask.U_ @ (e_st @ qqtTask.B.T + e_st_inv @ qqtTask.C.T) @ qqtTask.rootA0 @ qqtTask.R.T,
            ])

            right = left.T

            print('B shape: ', qqtTask.B.shape)
            print('e_st shape: ', e_st.shape)
            print('C shape: ', qqtTask.C.shape)
            print('lmda_inv shape: ', qqtTask.lmda_inv.shape)

            # TODO: instead of using lmda_inv im going to use s_inv
            # so dimensions match, I think this is what is meant in the paper

            # first = qqtTask.B @ (fractional_matrix_power(e_st, 2) - np.eye(qqtTask.out_dim)) @ s_inv @ qqtTask.B.T
            # print('first done: ', first)
            #
            # second = qqtTask.C @ (fractional_matrix_power(e_st, 2) - np.eye(qqtTask.out_dim)) @ s_inv @ qqtTask.C.T
            # print('second done: ', second)
            centre_centre = (
                    qqtTask.B @ (
                    fractional_matrix_power(e_st, 2) - np.eye(qqtTask.out_dim)) @ qqtTask.s_inv @ qqtTask.B.T
                    - qqtTask.C @ (fractional_matrix_power(e_st_inv, 2) - np.eye(
                qqtTask.out_dim)) @ qqtTask.s_inv @ qqtTask.C.T)

            print('R shape: ', qqtTask.R.shape)
            print('A0 shape: ', qqtTask.rootA0.shape)
            print('centre centre shape: ', centre_centre.shape)
            print('hidden_dim: ', qqtTask.hidden_dim)

            temp = qqtTask.R @ qqtTask.rootA0 @ centre_centre @ qqtTask.rootA0 @ qqtTask.R.T
            print('temp shape: ', temp.shape)
            centre = np.linalg.inv(np.eye(qqtTask.out_dim) +
                                   1 / 4 * qqtTask.R @ qqtTask.rootA0 @ centre_centre @ qqtTask.rootA0 @ qqtTask.R.T)

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))

            # TODO: A(0), RT i just made these the identity, not sure if they should be something else

    elif solution == '33':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            e_st = fractional_matrix_power(qqtTask.s, t / qqtTask.tau)
            e_st_inv = np.linalg.inv(e_st)

            left = 1 / 2 * np.vstack([
                qqtTask.V_ @ (e_st @ qqtTask.B.T - e_st_inv @ qqtTask.C.T) @ qqtTask.rootA0 @ qqtTask.R.T,
                qqtTask.U_ @ (e_st @ qqtTask.B.T + e_st_inv @ qqtTask.C.T) @ qqtTask.rootA0 @ qqtTask.R.T,
            ])

            centre_centre = (qqtTask.B @ (fractional_matrix_power(e_st, 2) - np.eye(qqtTask.in_dim))
                             @ qqtTask.s_inv @ qqtTask.B.T)

            centre_right = (qqtTask.C @ (fractional_matrix_power(e_st_inv, 2) - np.eye(qqtTask.in_dim))
                            @ qqtTask.s_inv @ qqtTask.C.T)

            centre = np.linalg.inv(
                np.linalg.inv(qqtTask.A0) + 1 / 4 * (centre_centre - centre_right)
            )

            right = left.T

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))

    elif solution == '37':
        for i in range(1, qqtTask.training_steps):
            t = i * qqtTask.learning_rate
            e_st = np.exp(qqtTask.s) ** t
            e_st_inv = np.linalg.inv(e_st)

            B_inv = np.linalg.inv(qqtTask.B)

            left = np.vstack([
                qqtTask.V_ @ (np.eye(qqtTask.out_dim)
                              - e_st_inv @ qqtTask.C.T @ np.linalg.inv(qqtTask.B).T @ e_st_inv),
                qqtTask.U_ @ (np.eye(qqtTask.out_dim)
                              - e_st_inv @ qqtTask.C.T @ np.linalg.inv(qqtTask.B).T @ e_st_inv)
            ])
            right = left.T
            centre_left = 4 * e_st_inv @ B_inv @ qqtTask.A0 @ B_inv.T @ e_st_inv
            centre_centre = (np.eye(qqtTask.s.shape[0]) - e_st_inv ** 2) @ np.linalg.inv(qqtTask.s)
            centre_right = - e_st_inv @ B_inv @ qqtTask.C @ (
                    fractional_matrix_power(e_st_inv, 2) - np.eye(qqtTask.in_dim)) @ np.linalg.inv(
                qqtTask.s) @ qqtTask.C.T @ B_inv.T @ e_st_inv

            centre = centre_left + centre_centre + centre_right

            next_val = left @ centre @ right
            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))

    # TODO: i should also return the losses
    # return QQts
    return np.asarray(losses), np.asarray(w2w1s)


def computationalSolution(qqtTask):
    # Now, generate the computational solution
    losses, ws = train_network(train=qqtTask.train,
                               learning_rate=qqtTask.learning_rate,
                               in_dim=qqtTask.in_dim,
                               hidden_dim=qqtTask.hidden_dim,
                               out_dim=qqtTask.out_dim,
                               init_w1=qqtTask.init_w1,
                               init_w2=qqtTask.init_w2,
                               training_steps=qqtTask.training_steps)

    return losses, ws


#randominputoutput mapping


def compareSolutions(solution):
    qqtTask = QQTTask(in_dim=8,
                      hidden_dim=8,
                      out_dim=8,
                      initial_scale=0.001,
                      batch_size=8,
                      learning_rate=0.001,
                      training_steps=200)

    analytical_ls, analytical_ws, = check_analytical_solution(solution, qqtTask)
    simulation_ls, simulation_ws = computationalSolution(qqtTask)

    # print('analytical shape: ', analytical_ws.shape)
    # print('simulation shape: ', simulation_ws.shape)
    print('simulation: ', simulation_ws[-3:])
    print('analytic: ', analytical_ws[-3:])

    """
    Plot Fobrenius norm in the difference between the two matrices
    """

    differences = [np.linalg.norm(simulation_w - analytical_w) for (simulation_w, analytical_w)
                   in zip(simulation_ws, analytical_ws)]

    # print(differences)
    #
    # plt.plot(differences)
    # plt.title('Norm of the differences between computational and analytical')
    # plt.show()
    """
    Plot trajectories of the representations of both
    """

    print('start simulation: ', simulation_ws[0])
    print('start analytic: ', analytical_ws[0])
    # print('end simulation: ', simulation_ws[-1])
    # print(simulation_ls)
    plt.plot(analytical_ws.reshape(analytical_ws.shape[0], -1), color=qqtTask.blind_colours[1], label = 'analytical')
    plt.legend()
    plt.figure()
    plt.plot(np.array(simulation_ws).reshape(np.array(simulation_ws).shape[0], -1),
              color=qqtTask.blind_colours[0], label='simulation')

    plt.legend()

    print('analytical loss: ', analytical_ls)
    print('simulation loss: ', simulation_ls)
    # plt.plot(analytical_ls)




    #plt.plot(analytical_ws.reshape(analytical_ws.shape[0], -1),
    #           c='k', alpha=0.7, linestyle=(0, (1,2)))

    plt.title('Analytical vs Simulation (4)')
    # plt.title('Losses Analytical')
    plt.ylabel('Network Output')
    plt.xlabel('Training Steps')
    plt.show()
    # rng = np.linspace(0.57, 1., 10)
    # # TODO: plot analytical output
    #
    # plt.figure()
    # # fig.set_xscale('log')
    #
    # plt.xlabel('Training Steps')
    # plt.title('Simulation Results')
    # # for color, output in zip(blind_colours, simulation_output[1].T):
    # #     for val in output:
    # #         plt.plot(rng, [val]*10, c=color, lw=2.5, clip_on=False, zorder=1)
    # #         plt.plot(rng, [val]*10, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), clip_on=False, zorder=2)
    # plot_matrix_evolution(simulation_output, 3)
    #
    # # TODO: plot simulation output
    # for n, (color) in enumerate(qqtTask.blind_colours[:qqtTask.plot_items_n]):
    #     plt.plot(-5, -5, c=color, lw=2.5, label=f"Item {n + 1}")
    # plt.plot(-5, -5, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label="Analytical")
    #
    # """
    # Plot difference between analytical and simulation
    # """
    # diff = np.array([m1 - m2 for m1, m2 in zip(analytical_output, simulation_output)])
    #
    # plt.figure()
    # plt.title('difference between simulation and analytical')
    # for color, output in zip(qqtTask.blind_colours, diff[1].T):
    #     for val in output:
    #         plt.plot(rng, [val] * 10, c=color, lw=2.5, clip_on=False, zorder=1)
    #         plt.plot(rng, [val] * 10, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), clip_on=False, zorder=2)
    #
    # plt.show()
    #
    # # print('analytical: ', analytical_output)
    # # print('simulation: ', simulation_output)
    # # print('diff: ', diff)
    return


equation_number = input('what equation should we check? Input a number: ')
compareSolutions(equation_number)
