import scipy.io
import os.path
import h5py
import mat73
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


data_dict_hier = mat73.loadmat('../Prior_local/data/Fig 4-Generalisaiton/General05-17-2022 12-488*8f/run.mat')

data_dict_55 = mat73.loadmat('../Prior_local/data/Fig 4-Generalisaiton/General05-17-2022 11-36-f7*7/run.mat')
data_dict_57 = mat73.loadmat('../Prior_local/data/Fig 4-Generalisaiton/General05-17-2022 12-265*7f/run.mat')
data_dict_75 = mat73.loadmat('../Prior_local/data/Fig 4-Generalisaiton/General05-17-2022 12-30-7*5f/run.mat')

sns.set_style("ticks", {
    'xtick.bottom': True,
    'xtick.top': False,
    'ytick.left': True,
    'ytick.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': '.1',
    'ytick.color': '.1',})
sns.set_context("talk")
hex_colors = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440"]
blind_colors = [mpl.colors.to_rgb(h) for h in hex_colors]
fig, axs = plt.subplots(1, 2, figsize=(7, 5.))
for i in range(2):
    if i==0:
        stds = data_dict_hier['gain_W1']
        std_max = data_dict_hier['l']
        outputs_8by8r = data_dict_hier['Error_RSA_W1_f']
        sigma = data_dict_hier['std_RSA_W1_f']
        X1_plus_sigma = outputs_8by8r + sigma
        X1_minus_sigma = outputs_8by8r - sigma
        k = 0
        for output, sigmin, sigplus,std in zip(outputs_8by8r, X1_minus_sigma, X1_plus_sigma,stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[0], lw=2.5,linestyle=(0, (1, 2)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[0])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[0], lw=2.5,)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[0])

        # # # # # # # #
        stds = data_dict_55['gain_W1']
        std_max = data_dict_55['l']
        outputs_5by5r = data_dict_55['Error_RSA_W1_f']
        sigma = data_dict_55['std_RSA_W1_f']
        X1_plus_sigma = outputs_5by5r + sigma
        X1_minus_sigma = outputs_5by5r - sigma
        k = 0
        for output, sigmin, sigplus,std in zip(outputs_5by5r, X1_minus_sigma, X1_plus_sigma,stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[1], lw=2.5,linestyle=(0, (1, 3)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[1])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[1], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2)
        # # # # # # # #
        stds =data_dict_75['gain_W1']
        std_max = data_dict_75['l']
        outputs_7by5r = data_dict_75['Error_RSA_W1_f']
        sigma = data_dict_75['std_RSA_W1_f']
        X1_plus_sigma = outputs_7by5r + sigma
        X1_minus_sigma = outputs_7by5r - sigma
        k = 0
        for output, sigmin, sigplus,std in zip(outputs_7by5r, X1_minus_sigma, X1_plus_sigma,stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[2], lw=2.5,linestyle=(0, (1, 4)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[2])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[2], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[2])
        # # # # # # # #
        stds = data_dict_57['gain_W1']
        std_max = data_dict_57['l']
        outputs_5by7r = data_dict_57['Error_RSA_W1_f']
        sigma = data_dict_57['std_RSA_W1_f']
        X1_plus_sigma = outputs_5by7r + sigma
        X1_minus_sigma = outputs_5by7r - sigma
        k = 0
        for output, sigmin, sigplus, std in zip(outputs_5by7r, X1_minus_sigma, X1_plus_sigma,stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[4], lw=2.5,linestyle=(0, (1, 5)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[4])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[4], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[4])
        axs[i].set_ylabel("$||W_1^TW_1-\~V\~S\~V^T||_F^2$")
        axs[i].yaxis.set_ticks([0,40,85])
        axs[i].set_xlim(0, 0.16)
        sns.despine(ax=axs[i])
        axs[i].xaxis.set_ticks([0, 0.15])
        axs[i].set_xlabel("Gain")

    # # # # # #
    else:
        stds = data_dict_hier['gain_W2']
        std_max = data_dict_hier['l']
        outputs_8by8r = data_dict_hier['Error_RSA_W2_f']
        sigma = data_dict_hier['std_RSA_W2_f']
        X1_plus_sigma = outputs_8by8r + sigma
        X1_minus_sigma = outputs_8by8r - sigma
        k = 0
        for output, sigmin, sigplus, std in zip(outputs_8by8r, X1_minus_sigma, X1_plus_sigma, stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[0], lw=2.5,linestyle=(0, (1, 2)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[0])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[0], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[0])

        # # # # # # # #
        stds = data_dict_55['gain_W2']
        std_max = data_dict_55['l']
        outputs_5by5r = data_dict_55['Error_RSA_W2_f']
        sigma = data_dict_55['std_RSA_W2_f']
        X1_plus_sigma = outputs_5by5r + sigma
        X1_minus_sigma = outputs_5by5r - sigma
        k = 0
        for output, sigmin, sigplus, std in zip(outputs_5by5r, X1_minus_sigma, X1_plus_sigma, stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[1], lw=2.5,linestyle=(0, (1, 3)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[1])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[1], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[1])
        # # # # # # # #
        stds = data_dict_75['gain_W2']
        std_max = data_dict_75['l']
        outputs_7by5r = data_dict_75['Error_RSA_W2_f']
        sigma = data_dict_75['std_RSA_W2_f']
        X1_plus_sigma = outputs_7by5r + sigma
        X1_minus_sigma = outputs_7by5r - sigma
        k = 0
        for output, sigmin, sigplus, std in zip(outputs_7by5r, X1_minus_sigma, X1_plus_sigma, stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[2], lw=2.5,linestyle=(0, (1, 4)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[3])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[2], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[2])
        # # # # # # # #
        stds = data_dict_57['gain_W2']
        std_max = data_dict_57['l']
        outputs_5by7r = data_dict_57['Error_RSA_W2_f']
        sigma = data_dict_57['std_RSA_W2_f']
        X1_plus_sigma = outputs_5by7r + sigma
        X1_minus_sigma = outputs_5by7r - sigma
        k = 0
        for output, sigmin, sigplus, std in zip(outputs_5by7r, X1_minus_sigma, X1_plus_sigma, stds):
            k = k + 1
            if k == 1:
                axs[i].plot(std, output, c=blind_colors[4], lw=2.5,linestyle=(0, (1, 5)))
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[4])
            if k == 2:
                axs[i].plot(std, output, c=blind_colors[4], lw=2.5)
                axs[i].fill_between(std, sigplus, sigmin, alpha=0.2, color=blind_colors[4])
        axs[i].set_xlabel("Gain")
        axs[i].set_xlim(0, 0.16)
        axs[i].yaxis.set_ticks([0,40,85])
        axs[i].xaxis.set_ticks([0,0.15])
        sns.despine(ax=axs[i])
        axs[i].set_ylabel("$||W_2 W_2^T-\~U^T\~S\~U||_F^2$")


fig.tight_layout()
fig.subplots_adjust(right=0.85)
fig.savefig("./figures/figure-4-Generalisation.svg")
fig.savefig("./figures/figure-4-Generalisation.pdf")

if __name__ == '__main__':
    print('hello')