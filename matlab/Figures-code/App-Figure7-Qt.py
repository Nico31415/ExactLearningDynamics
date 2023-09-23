import scipy.io
import os.path
import h5py
import mat73
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data_dict = mat73.loadmat('../Prior_local/data/Test-Q/Test03-08-2022 09-49_2/run.mat')

sns.set_style("ticks", {
    'xtick.bottom': True,
    'xtick.top': False,
    'ytick.left': True,
    'ytick.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': '.1',
    'ytick.color': '.1',
})

sns.set_context("talk")
hex_colors = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440"]
blind_colors = [mpl.colors.to_rgb(h) for h in hex_colors]
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
for i in range(2):
    for j in range(4):
            if i == 0:
                if j == 0:
                    training_steps_1 = data_dict['t1']
                    t3= data_dict['t3']
                    trainning_step_max = np.max(t3)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 10000)])
                    outputs_2 = data_dict['losses']  # My data array
                    axs[i,j].plot(outputs_2, color=blind_colors[1], lw=2.5, label='Task 1')
                    outputs_3 = data_dict['losses_tilde']
                    axs[i,j].plot(outputs_3, color=blind_colors[3], lw=2.5, label='Task 2')
                    sns.despine(ax=axs[i, j])
                    axs[i, j].axvline(x=training_steps_1, ymin=0.0, ymax=np.max(outputs_2) + 1000, color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=4)
                    axs[i,j].set_yticks([0, 1.25e-4])
                    axs[i, j].set_ylabel("Loss")

                if j == 1:
                    training_steps_1= data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['QtQtTs_2']  # My data array
                    analytical = data_dict['Qqt']
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i,j].plot(val, c=blind_colors[1], lw=2.5)
                    for color, analytical in zip(blind_colors, analytical.T):
                        for val in analytical:
                            axs[i,j].plot(val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                    axs[i,j].set_xlabel("Training Steps")
                    axs[i,j].set_xlim(1, training_steps_1 )
                    sns.despine(ax=axs[i,j])
                    axs[i,j].set_ylabel("$QQ(t)^T$")
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs) + 1000,
                                      color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=4)
                    axs[i,j].set_yticks([0, 1e-2])

                if j == 2:
                    training_steps_1 = data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['Q_2']  # My data array
                    analytical = data_dict['Qt']
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i,j].plot(val, c=blind_colors[1], lw=2.5)
                    for color, analytical in zip(blind_colors, analytical.T):
                        for val in analytical:
                            axs[i,j].plot(val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                    axs[i,j].set_xlabel("Training Steps")
                    axs[i,j].set_ylabel("$q(t)$")
                    axs[i,j].set_xlim(1, training_steps_1)
                    sns.despine(ax=axs[i,j])
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs) + 1000,
                                      color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=6)
                    axs[i,j].set_yticks([0, 1e-1])

                if j == 3:
                    training_steps_1 = data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['QtQtTs_2']  # My data array
                    analytical = data_dict['QQ_2']
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i, j].plot(val, c=blind_colors[1], lw=2.5)
                    for color, analytical in zip(blind_colors, analytical.T):
                        for val in analytical:
                            axs[i, j].plot(val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                    axs[i, j].set_xlabel("Training Steps")
                    axs[i, j].set_xlim(1, training_steps_1)
                    sns.despine(ax=axs[i, j])
                    axs[i, j].set_ylabel("$q(t)q(t)^T$")
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs) + 1000,
                                      color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=4)
                    axs[i, j].set_yticks([0, 1e-2])

            if i == 1:
                if j == 2:
                    training_steps_1 = data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['D']  # My data array
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i,j].plot(val, c=blind_colors[1], lw=2.5)
                    axs[i,j].set_xlabel("Training Steps")
                    axs[i,j].set_xlim(1, training_steps_1)
                    sns.despine(ax=axs[i,j])
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs_3) + 1000,
                                      color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=6)
                    axs[i, j].set_yticks([-2,0, 2])
                    axs[i, j].set_ylabel("$D(t)$")

                if j == 0:
                    training_steps_1 = data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['Qt_d']  # My data array
                    analytical = data_dict['Q_2']
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i,j].plot(-val, c=blind_colors[1], lw=2.5)
                    for color, analytical in zip(blind_colors, analytical.T):
                        for val in analytical:
                            axs[i,j].plot(val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                    axs[i,j].set_xlabel("Training Steps")
                    axs[i,j].set_xlim(1, training_steps_1)
                    sns.despine(ax=axs[i,j])
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs_3) + 1000,
                                      color=blind_colors[2],
                                      linestyle=(0, (1, 2)), linewidth=6)
                    axs[i, j].set_yticks([0, 1e-1])
                    axs[i, j].set_ylabel("$Q_d(t)$")

                if j == 1:
                    training_steps_1 = data_dict['t3']
                    trainning_step_max = np.max(training_steps_1)
                    axs[i, j].xaxis.set_ticks(
                        [0, trainning_step_max - np.mod(trainning_step_max, 100000)])
                    outputs = data_dict['QtQtTs_2']  # My data array
                    analytical = data_dict['QQ_2_d']
                    for color, output in zip(blind_colors, outputs.T):
                        for val in output:
                            axs[i,j].plot(val, c=blind_colors[1], lw=2.5)
                    for color, analytical in zip(blind_colors, analytical.T):
                        for val in analytical:
                            axs[i,j].plot(val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                    axs[i,j].set_xlabel("Training Steps")
                    axs[i,j].set_xlim(1, training_steps_1)
                    sns.despine(ax=axs[i,j])
                    axs[i, j].axvline(x=0, ymin=0.0, ymax=np.max(outputs_3) + 1000,
                                  color=blind_colors[2],
                                  linestyle=(0, (1, 2)), linewidth=6)
                    axs[i, j].set_yticks([0, 1e-2])
                    axs[i, j].set_ylabel("$Q_d(t)Q_d^T(t)$")

fig.legend(loc=7, fontsize=14, frameon=False)
fig.tight_layout()
fig.subplots_adjust(right=0.85)
fig.savefig("./figures/figure-app-Q.svg")
fig.savefig("./figures/figure-app-Q.pdf")
if __name__ == '__main__':
    print('hello')