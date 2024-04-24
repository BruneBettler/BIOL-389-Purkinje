# Author: Brune Bettler
# Date: April 4th 2024

import pandas as pd
from posterior_analysis import get_stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, kstest, mannwhitneyu, ttest_ind

def frequency_box(ant_freq_data, post_freq_data):
    '''
    mean of all cells both ant and post together in one plot
    '''

    box1 = plt.boxplot(ant_freq_data, positions=[1], patch_artist=True)
    box2 = plt.boxplot(post_freq_data, positions=[2], patch_artist=True)

    box1['boxes'][0].set_facecolor('red')
    box2['boxes'][0].set_facecolor('blue')
    plt.xticks([1, 2], ['Anterior', 'Posterior'])

    plt.ylabel('Spike-Train Frequency (Hz)')

    stat, p = mannwhitneyu(ant_freq_data, post_freq_data)
    plt.title(f'Anterior vs Posterior Frequency BoxPlots \n All Spike-Trains | p = {round(p, 3)}')

    plt.savefig('ant vs post spike-train boxplots.png')

    plt.tight_layout()

    plt.show()

    return 0

def CV_box(ant_CV_data, post_CV_data):
    box1 = plt.boxplot(ant_CV_data, positions=[1], patch_artist=True)
    box2 = plt.boxplot(post_CV_data, positions=[2], patch_artist=True)

    box1['boxes'][0].set_facecolor('red')
    box2['boxes'][0].set_facecolor('blue')
    plt.xticks([1, 2], ['Anterior', 'Posterior'])

    plt.ylabel('CV')

    stat, p = mannwhitneyu(ant_CV_data, post_CV_data)
    #p = "{:.2e}".format(p)
    plt.title(f'Anterior vs Posterior CV BoxPlots \n All Cells | p = {round(p, 4)}')

    #plt.savefig('CV ant vs post cells boxplots.png')

    plt.tight_layout()

    plt.show()

    print(p)

    return 0

def compare_cells_to_trains(ant_cell_mean_CV, post_cell_mean_CV, ant_spike_CVs, post_spike_CVs, ant_cell_mean_freq, post_cell_mean_freq, ant_spike_freqs, post_spike_freqs):
    # make two plots

    # first plot = CV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), sharey=True)
    fig.subplots_adjust(top=0.85)
    plt.suptitle('CV per Cell vs CV per Spike-Train')

    # two subplots right ant red left post blue

    # Right ant red
    box1 = ax1.boxplot(ant_cell_mean_CV, positions=[1], patch_artist=True)
    box2 = ax1.boxplot(ant_spike_CVs, positions=[2], patch_artist=True)
    box1['boxes'][0].set_facecolor('red')
    box2['boxes'][0].set_facecolor('red')
    ax1.set_xticks([1, 2], ['Per Cell', 'Per Spike Train'])
    ax1.set_ylabel('CV')

    stat, p = mannwhitneyu(ant_cell_mean_CV, ant_spike_CVs)
    t_stat, t_p = ttest_ind(ant_cell_mean_freq, ant_spike_freqs)
    ax1.set_title(f'Anterior \nM.Whit. p={round(p, 2)}  |  t-test p={round(t_p, 2)}')


    # left post blue
    box1 = ax2.boxplot(post_cell_mean_CV, positions=[1], patch_artist=True)
    box2 = ax2.boxplot(post_spike_CVs, positions=[2], patch_artist=True)
    box1['boxes'][0].set_facecolor('blue')
    box2['boxes'][0].set_facecolor('blue')
    ax2.set_xticks([1, 2], ['Per Cell', 'Per Spike Train'])

    stat, p = mannwhitneyu(post_cell_mean_CV, post_spike_CVs)
    t_stat, t_p = ttest_ind(post_cell_mean_freq, post_spike_freqs)
    ax2.set_title(f'Posterior \nM.Whit. p={round(p, 2)}  |  t-test p={round(t_p, 2)}')

    plt.tight_layout()
    #plt.savefig('CV cell vs train.png')
    #plt.show()

    ''' ------------------------------------------------------------------------------------- '''
    # second plot = Freq

    '''# two subplots right ant red left post blue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), sharey=True)
    fig.subplots_adjust(top=0.85)
    plt.suptitle('Frequency per Cell vs Frequency per Spike-Train')

    # Right ant red
    box1 = ax1.boxplot(ant_cell_mean_freq, positions=[1], patch_artist=True)
    box2 = ax1.boxplot(ant_spike_freqs, positions=[2], patch_artist=True)
    box1['boxes'][0].set_facecolor('red')
    box2['boxes'][0].set_facecolor('red')
    ax1.set_xticks([1, 2], ['Per Cell', 'Per Spike Train'])
    ax1.set_ylabel('Frequency (Hz)')

    stat, p = mannwhitneyu(ant_cell_mean_freq, ant_spike_freqs)
    t_stat, t_p = ttest_ind(ant_cell_mean_freq, ant_spike_freqs)
    ax1.set_title(f'Anterior \nM.Whit. p={round(p, 2)}  |  t-test p={round(t_p, 2)}')

    # left post blue
    box1 = ax2.boxplot(post_cell_mean_freq, positions=[1], patch_artist=True)
    box2 = ax2.boxplot(post_spike_freqs, positions=[2], patch_artist=True)
    box1['boxes'][0].set_facecolor('blue')
    box2['boxes'][0].set_facecolor('blue')
    ax2.set_xticks([1, 2], ['Per Cell', 'Per Spike Train'])

    stat, p = mannwhitneyu(post_cell_mean_freq, post_spike_freqs)
    t_stat, t_p = ttest_ind(post_cell_mean_freq, post_spike_freqs)
    ax2.set_title(f'Posterior \nM.Whit. p={round(p, 2)}  |  t-test p={round(t_p, 2)}')

    plt.tight_layout()
    plt.savefig('FREQ cell vs train.png')
    plt.show()'''

    return 0


if __name__ == '__main__':
    post_data = pd.read_csv('data/posterior_data.csv')
    post_cell_indices = [(0, 9), (11, 20), (22, 30), (32, 38), (40, 44), (46, 52), (54, 61), (63, 71)]

    ant_data = pd.read_csv('data/anterior_data.csv')
    ant_cell_indices = [(0, 8), (10, 19), (21, 27), (29, 38), (40, 49), (51, 60), (62, 71), (73, 82), (84, 93), (95, 104)]

    ant_cells_freq = []
    ant_cells_CV = []
    spike_ant_cells_freq = []
    spike_ant_cells_CV = []

    for i, indices in enumerate(ant_cell_indices):
        ant_cells_freq.append(get_stats(ant_data, indices)[1])
        spike_ant_cells_freq.append(get_stats(ant_data, indices)[3])
        ant_cells_CV.append(get_stats(ant_data, indices)[2])
        spike_ant_cells_CV.append(get_stats(ant_data, indices)[4])

    post_cells_freq = []
    post_cells_CV = []
    spike_post_cells_freq = []
    spike_post_cells_CV = []
    for i, indices in enumerate(post_cell_indices):
        post_cells_freq.append(get_stats(post_data, indices)[1])
        spike_post_cells_freq.append(get_stats(post_data, indices)[3])
        post_cells_CV.append(get_stats(post_data, indices)[2])
        spike_post_cells_CV.append(get_stats(post_data, indices)[4])

    spike_ant_cells_F = [item for sublist in spike_ant_cells_freq for item in sublist]
    spike_ant_cells_C = [item for sublist in spike_ant_cells_CV for item in sublist]
    spike_post_cells_F = [item for sublist in spike_post_cells_freq for item in sublist]
    spike_post_cells_C = [item for sublist in spike_post_cells_CV for item in sublist]

    compare_cells_to_trains(ant_cells_CV, post_cells_CV, spike_ant_cells_C, spike_post_cells_C, ant_cells_freq, post_cells_freq, spike_ant_cells_F, spike_post_cells_F)


