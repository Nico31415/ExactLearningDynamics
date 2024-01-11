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


# generates the computational solution using gradient descent
def train_network(train, learning_rate, in_dim, hidden_dim, out_dim, init_w1, init_w2, training_steps):
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

    ws = np.zeros((training_steps, in_dim, out_dim))
    ws[0] = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]
    # ws = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]




    for training_step in range(training_steps - 1):
        print('step: ', training_step)
        state, params, current_loss = trainer(state, params)
        losses.append(current_loss)
        # ws.append(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])
        print(params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])
        ws[training_step+1] = params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"]
        # np.append(ws, params["network"]["layer-1"]["w"] @ params["network"]["layer-0"]["w"])

    return losses, ws


def check_analytical_solution(solution):
    # takes in solution (string) number on paper "Follow Up Deep Linear Network"
    """
    Checks analytical solution from paper against computational result.
    Input: number of equation on the paper

    3: 1/tau (FQQt + QQtF - (QQt)^2)

    4: exponent stuff, doesn't converge well
    """

    qqtTask = QQTTask(in_dim=3,
                      hidden_dim=4,
                      out_dim=3,
                      initial_scale=0.35,
                      batch_size=10,
                      learning_rate=0.1,
                      training_steps=400)

    QQts = [qqtTask.qqt0]
    w2w1s = [qqtTask.getw2w1(qqtTask.qqt0)]

    if solution == '3':
        for i in range(1, qqtTask.training_steps):
            curr = QQts[-1]

            derivative = qqtTask.F @ curr + curr @ qqtTask.F - curr @ curr.T
            next_val = curr + qqtTask.learning_rate * derivative
            QQts.append(next_val)

            w2w1s.append(qqtTask.getw2w1(next_val))
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

    elif solution == '4':
        for i in range(1, qqtTask.training_steps):
            tau = 1
            t = i * qqtTask.learning_rate
            e_ft = np.exp(t / tau * qqtTask.F)

            out = e_ft @ qqtTask.q0
            centre_centre = e_ft @ np.linalg.inv(qqtTask.F) @ e_ft - np.linalg.inv(qqtTask.F)
            # print(np.eye(F.shape[0]) + 1/2 * q0.T @ centre_centre @ q0)
            try:
                centre = np.linalg.inv(np.eye(qqtTask.F.shape[0]) + 1 / 2 * qqtTask.q0.T @ centre_centre @ qqtTask.q0)
            except:
                print(i)
                return
            next_val = out @ centre @ out.T

            QQts.append(next_val)
            w2w1s.append(qqtTask.getw2w1(next_val))
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))
        return

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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))
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
            # w2w1s.append(np.array([next_val[-required_shape[0]:, :required_shape[1]]]))

    # Now, generate the computational solution
    losses, ws = train_network(train= qqtTask.train,
                               learning_rate= qqtTask.learning_rate,
                               in_dim=qqtTask.in_dim,
                               hidden_dim=qqtTask.hidden_dim,
                               out_dim = qqtTask.out_dim,
                               init_w1= qqtTask.init_w1,
                               init_w2= qqtTask.init_w2,
                               training_steps= qqtTask.training_steps)

    print('analytic weights: ', w2w1s[:3])
    print('simulation weights: ', ws)

    # i don't understand this, why dont i just do it for each one
    # analytical_output = (np.asarray(w2w1s)[:, 0, :, :] @ qqtTask.X[:qqtTask.plot_items_n].T)
    # simulation_output = (np.asarray(ws)[:, 0, :, :] @ qqtTask.X[:qqtTask.plot_items_n].T)

    analytical_output = np.asarray([np.asarray(w2w1) @ qqtTask.X.T for w2w1 in w2w1s])
    simulation_output = np.asarray([np.asarray(w) @ qqtTask.X.T for w in ws])

    print('analytical shape: ', analytical_output.shape)
    print('simulation shape: ', simulation_output.shape)

    # print('simulation: ', simulation_output[:3])
    #
    # print('analytic: ', analytical_output[:3])
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
    for n, (color) in enumerate(qqtTask.blind_colours[:qqtTask.plot_items_n]):
        plt.plot(-5, -5, c=color, lw=2.5, label=f"Item {n + 1}")
    plt.plot(-5, -5, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label="Analytical")

    """
    Plot difference between analytical and simulation
    """
    diff = np.array([m1 - m2 for m1, m2 in zip(analytical_output, simulation_output)])

    plt.figure()
    plt.title('difference between simulation and analytical')
    for color, output in zip(qqtTask.blind_colours, diff[1].T):
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
