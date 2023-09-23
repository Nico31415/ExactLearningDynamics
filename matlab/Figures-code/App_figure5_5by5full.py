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
data_dict_large_equal= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-37-app-large/run.mat')
data_dict_inter_equal=mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-21-app-inter/run.mat')
data_dict_small_equal= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-20-app-small/run.mat')
#diagonal_unequal
data_dict_large_diag=mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-28_app_large/run.mat')
data_dict_inter_diag=mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-22_app_inter/run.mat')
data_dict_small_diag= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-31_app_small/run.mat')
# dense
data_dict_large= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-43-app-large/run.mat')
data_dict_inter=mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-44-app-inter/run.mat')
data_dict_small= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-23-2022 22-47-app-small/run.mat')

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
#009E73”, “#CC78A6

hex_colors = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440"]
blind_colors = [mpl.colors.to_rgb(h) for h in hex_colors]
cmap = mpl.colors.ListedColormap(hex_colors)
fig, axs = plt.subplots(3, 3, figsize=(10, 8))
for i in range(3):
    for j in range(3):
        if i == 0:
            if j == 0:
                outputs = data_dict_small['W2W1_ij']  # My data array
                trainning_step= data_dict_small['v']
                t_bump = data_dict_small['t_bumps']
                trainning_step_max=np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i,j].set_ylabel("$W_2W_1(t)$")
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

            if j == 1:
                outputs = data_dict_inter['W2W1_ij']  # My data array
                trainning_step = data_dict_inter['v']
                trainning_step_max = np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)

                axs[i, j].tick_params(axis='both', which='major', labelsize=12)



            if j == 2:
                outputs = data_dict_large['W2W1_ij']  # My data array
                trainning_step= data_dict_large['v']
                k=0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)


                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

        if i == 1:
            if j == 0:
                outputs = data_dict_small_diag['W2W1_ij']  # My data array
                trainning_step= data_dict_small_diag['v']
                t_bump= data_dict_small_diag['t_bumps']
                k=0
                trainning_step_max = np.max(trainning_step)
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i,j].set_ylabel("$W_2W_1(t)$")

            if j == 1:
                outputs = data_dict_inter_diag['W2W1_ij']  # My data array
                trainning_step = data_dict_inter_diag['v']
                trainning_step_max = np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)


            if j == 2:
                outputs = data_dict_large_diag['W2W1_ij']  # My data array
                trainning_step= data_dict_large_diag['v']
                k=0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

        if i == 2:
            if j == 0:
                outputs = data_dict_small_equal['W2W1_ij']  # My data array
                trainning_step= data_dict_small_equal['v']
                t_bump = data_dict_small_equal['t_bumps']
                trainning_step_max=np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)
                axs[i,j].set_ylabel("$W_2W_1(t)$")
            if j == 1:
                outputs = data_dict_inter_equal['W2W1_ij']  # My data array
                trainning_step = data_dict_inter_equal['v']
                trainning_step_max = np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

            if j == 2:
                outputs = data_dict_large_equal['W2W1_ij']  # My data array
                trainning_step = data_dict_large_equal['v']
                trainning_step_max = np.max(trainning_step)
                k = 0
                for color, output in zip(blind_colors, outputs.T):
                    for val in output:
                        k = k + 1
                        if k == 1 or k == 13 or k == 7 or k == 19 or k == 25:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[1], lw=2.5)
                        else:
                            axs[i, j].plot(trainning_step, val, c=blind_colors[0], lw=2.5)
                axs[i, j].set_xlabel('Training steps')
                axs[i, j].tick_params(axis='both', which='major', labelsize=12)

pad = 7
cols = ['Small','Intermediate','Large']
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')

fig.tight_layout()
fig.subplots_adjust(right=0.85)
fig.savefig("./figures/figure-5-5by5full.svg")
fig.savefig("./figures/figure-5-5by5full.pdf")

if __name__ == '__main__':
    print('hello')
