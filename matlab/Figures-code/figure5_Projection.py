import scipy.io
import os.path
import h5py
import mat73
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import numpy as np
#diagonal_equal
data_dict_large_equal= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 12-20_large/run.mat')
data_dict_small_equal= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 12-06_small/run.mat')
data_dict_inter_equal=mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 19-49_inter_2/run.mat')
#diagonal_unequal
data_dict_small_diag= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 13-34_small_diag/run.mat')
data_dict_inter_diag=mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 13-24_inter_diag/run.mat')
data_dict_large_diag= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 12-23_large_diag/run.mat')
data_dict_inter_diag=mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 20-01_inter_2_diag/run.mat')
# dense
data_dict_large= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 18-39_large_full/run.mat')
data_dict_inter=mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 18-28_inter/run.mat')
data_dict_small= mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-16-2022 00-40_small/run.mat')
data_dict_inter=mat73.loadmat('../Prior_local/data/Fig5_Alignement/AA05-15-2022 19-24_inter_2/run.mat')


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
mpl.rcParams.update({'font.size': 9})
sns.set_context("talk")
hex_colors = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440"]
blind_colors = [mpl.colors.to_rgb(h) for h in hex_colors]
cmap = mpl.colors.ListedColormap(hex_colors)
fig, axs = plt.subplots(3, 3, figsize=(10, 8))
for i in range(3):
    for j in range(3):
        if i == 0:
            if j == 0:
                outputs = data_dict_small['Projection_W2W1_ij']  # My data array
                analitical = data_dict_small['Projection_QQW2W1_ij']
                trainning_step= data_dict_small['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 100000)])
                k = 1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1;
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                                t_bump_num = np.where(val == np.amax(val))
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].set_ylabel("$A^TA(t)$")
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3],
                                  linestyle=(0, (1, 2)), linewidth=4)

            if j == 1:
                outputs = data_dict_inter['Projection_W2W1_ij']  # My data array
                analitical = data_dict_inter['Projection_QQW2W1_ij']
                trainning_step = data_dict_inter['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 100000)])
                k = 1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1;
                        if k == 2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k == 5:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                            t_bump_num = np.where(val == np.amax(val))
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step, val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3],
                                  linestyle=(0, (1, 2)),linewidth=4)

                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

            if j == 2:
                outputs = data_dict_large['Projection_W2W1_ij']  # My data array
                analitical = data_dict_large['Projection_QQW2W1_ij']
                trainning_step= data_dict_large['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 10000)])
                k=1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1;
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                        t_bump_num = np.where(val == np.amax(val))

                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].yaxis.set_ticks([0, 30, 60])



        if i == 1:
            if j == 0:
                outputs = data_dict_small_diag['Projection_W2W1_ij']  # My data array
                analitical = data_dict_small_diag['Projection_QQW2W1_ij']
                trainning_step= data_dict_small_diag['v']

                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 100000)])
                k=1
                trainning_step_max = np.max(trainning_step)

                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                                t_bump_num = np.where(val == np.amin(val))
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].set_ylabel("$A^TA(t)$")
                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3],
                                  linestyle=(0, (1, 2)), linewidth=4)


            if j == 1:
                outputs = data_dict_inter_diag['Projection_W2W1_ij']  # My data array
                analitical = data_dict_inter_diag['Projection_QQW2W1_ij']
                trainning_step = data_dict_inter_diag['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 10000)])
                k = 1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1;
                        if k == 2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k == 5:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                            t_bump_num = np.where(val == np.amax(val))

                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step, val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))

                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3], linestyle=(0, (1, 2)),linewidth=4)
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)


            if j == 2:
                outputs = data_dict_large_diag['Projection_W2W1_ij']  # My data array
                analitical = data_dict_large_diag['Projection_QQW2W1_ij']
                trainning_step= data_dict_large_diag['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 10000)])
                k=1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)

                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].yaxis.set_ticks([0, 30, 60])

        if i == 2:
            if j == 0:
                outputs = data_dict_small_equal['Projection_W2W1_ij']  # My data array
                analitical = data_dict_small_equal['Projection_QQW2W1_ij']
                trainning_step= data_dict_small_equal['v']
                t_bump = data_dict_small_equal['t_bumps']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 10000)])

                k = 1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1;
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                                t_bump_num = np.where(val == np.amax(val))

                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].set_ylabel("$A^TA(t)$")
                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3],
                                  linestyle=(0, (1, 2)), linewidth=4)
                axs[i, j].axvline(x=t_bump, ymin=0.0, ymax=trainning_step_max, color=blind_colors[2],
                                  linestyle=(0, (1, 2)),linewidth=4)
            if j == 1:
                outputs = data_dict_inter_equal['Projection_W2W1_ij']  # My data array
                analitical = data_dict_inter_equal['Projection_QQW2W1_ij']
                t_bump = data_dict_inter_equal['t_bumps']
                trainning_step = data_dict_inter_equal['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 10000)])
                k = 1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1;
                        if k == 2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k == 5:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                            t_bump_num = np.where(val == np.amax(val))
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step, val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].axvline(x=t_bump_num, ymin=0.0, ymax=trainning_step_max, color=blind_colors[3],
                                  linestyle=(0, (1, 2)),linewidth=4)
                axs[i, j].axvline(x=t_bump, ymin=0.0, ymax=trainning_step_max, color=blind_colors[2],
                                  linestyle=(0, (1, 2)), linewidth=4)
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

            if j == 2:
                outputs = data_dict_large_equal['Projection_W2W1_ij']  # My data array
                analitical = data_dict_large_equal['Projection_QQW2W1_ij']
                trainning_step= data_dict_large_equal['v']
                trainning_step_max = np.max(trainning_step)
                axs[i, j].xaxis.set_ticks(
                    [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 1000)])
                k=1
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k=k+1;
                        if k==2:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        elif k==5:
                                axs[i, j].plot(trainning_step,val, c=blind_colors[1], lw=2.5)
                        else:
                                axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                for color, analitical in zip(blind_colors, analitical.T):
                    for val in analitical:
                        axs[i, j].plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i, j].yaxis.set_ticks([0, 30, 60])
pad = 7
cols = ['Small','Intermediate','Large']
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')

fig.tight_layout()
fig.subplots_adjust(right=0.85)
sns.despine()
fig.savefig("./figures/figure-5-projw2w1.svg")
fig.savefig("./figures/figure-5-projw2w1.pdf")

if __name__ == '__main__':
    print('hello')