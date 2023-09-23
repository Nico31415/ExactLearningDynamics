import scipy.io
import os.path
import h5py
import mat73
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import numpy as np

#diagonal_unequal
data_dict_inter= mat73.loadmat('../Prior_local/data/Fig5_Alignement/fivefive05-08-2022 18-33/run.mat')

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
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
analitical=  data_dict_inter['Qqt_square_W2W1']
outputs = data_dict_inter['W2W1_ij']  # My data array
trainning_step= data_dict_inter['v']
t_bump = data_dict_inter['t_bumps']
trainning_step_max=np.max(trainning_step)
k = 0
for color, output in zip(blind_colors, outputs.T):
    for val in output:
        k=k+1
        if k==1 :
            axs.plot(trainning_step, val, c=blind_colors[1], lw=2.5, label='Diagonal')
        elif k==13 or  k==7 or  k==19 or  k == 25:
            axs.plot(trainning_step, val, c=blind_colors[1], lw=2.5)
        elif  k==14:
            axs.plot(trainning_step, val, c=blind_colors[0], lw=2.5, label='Off-Diagonal')
        else:
            axs.plot(trainning_step, val, c=blind_colors[0], lw=2.5)
for color, analitical in zip(blind_colors, analitical.T):
        for val in analitical:
            axs.plot(trainning_step,val, lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)))
        axs.set_ylabel("$W_2W_1(t)$")
        axs.tick_params(axis='both', which='major', labelsize=12)
        axs.set_xlabel('Training steps')
        #axs.axvline(x=t_bump, ymin=0.0, ymax=trainning_step_max, color=blind_colors[2],linestyle=(0, (1, 2)), linewidth=4)
        axs.xaxis.set_ticks(
                            [np.min(trainning_step), trainning_step_max - np.mod(trainning_step_max, 100)])
        axs.yaxis.set_ticks([0, 3, 6])


fig.tight_layout()
sns.despine()
fig.subplots_adjust(right=0.85)
fig.savefig("./figures/figure-5-5by5.svg")
fig.savefig("./figures/figure-5-5by5.pdf")
if __name__ == '__main__':
    print('hello')