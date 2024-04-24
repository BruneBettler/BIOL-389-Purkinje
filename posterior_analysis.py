# Author: Brune Bettler
# Date: April 3rd 2024

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
cell # info = row index 0
spike time (seconds) = row indices 2 - 652 
total_cell_num = 8 
each train is 8 seconds 
'''

def get_train(data, train_num, cell_indicies):
    start_col = cell_indicies[0]

    column_data = data.iloc[2:, start_col + train_num]

    clean_data = []

    for spike_time in column_data:
        spike_time = float(spike_time)
        if not np.isnan(spike_time):
            clean_data.append(spike_time)
        else:
            break

    return np.array(clean_data)


def get_col_freq(column_data):
    column_data = np.array(column_data)
    intervals = np.diff(column_data)
    frequency = np.mean(1/intervals)

    return frequency

def cell_freq_per_train(data, cell_indicies, cell_num):
    start_col = cell_indicies[0]
    end_col = cell_indicies[1]
    train_num = end_col - start_col

    all_mean_freq = []
    all_CV = []

    min_y = 0
    max_y = 0

    blues = ['#4D4DFF', '#4545E6', '#3D3DCC', '#3535B3', '#2D2D99', '#252580', '#1D1D66', '#15154D', '#0D0D33', '#05051A']

    # make big graph with correct number of subgraphs
    fig, axs = plt.subplots(nrows=train_num, ncols=1, figsize=(8, 12))

    for train_id in range(train_num):
        train_data = get_train(data, train_id, cell_indicies)

        spike_num = len(train_data) # add this to graph n = __

        intervals = np.diff(train_data)
        CV = np.std(intervals) / np.mean(intervals)
        all_CV.append(CV)

        min = np.min(intervals)
        max = np.max(intervals)

        if min < min_y: min_y = min
        if max > max_y: max_y = max

        mean_inst_freq = np.mean((1/intervals)) # add this to graph
        all_mean_freq.append(mean_inst_freq)

        midpoint_time = (train_data[:-1] + train_data[1:]) / 2

        #slope, intercept, r_value, p_value, std_err = linregress(midpoint_time, intervals)
        #print(f"Slope: {slope}")
        #print(f"P-value: {p_value}")
        #axs[train_id].plot(time, intercept + slope * time, 'r', label='Fitted line')

        # Plotting the data and the regression line
        axs[train_id].scatter(midpoint_time, intervals, label='Data', color=blues[train_id])
        #axs[train_id].legend()
        #axs[train_id].set_title('Train ' + str(train_id+1), fontsize=10)

        # Annotation for the number of cells and frequency
        annotation_text = f'n = {spike_num} \n{round(mean_inst_freq,1)} Hz \nCV: {round(CV, 2)}'
        # Place text in the top-left corner of the subplot
        # Adjust x and y for different corners or positions
        axs[train_id].text(0.90, 0.90, annotation_text, va='top', ha='left', transform=axs[train_id].transAxes, fontsize=8)

    # show graph

    mean_freq = np.mean(np.array(all_mean_freq))
    mean_CV = np.mean(all_CV)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Spike Timing Intervals Over Time \n Posterior Cell ' + str(cell_num+1) + ' | ' + str(train_num) + ' Spike Trains | Mean Frequency: ' + str(round(mean_freq, 1)) + "Hz | Mean CV: " + str(round(mean_CV, 2)))
    fig.subplots_adjust(hspace=0)  # hspace=0 makes the plots touch each other
    fig.subplots_adjust(left=0.15)  # Increase the left margin
    fig.text(0.04, 0.5, 'Individual Spike Timing Interval (seconds)', va='center', ha='center', rotation='vertical', fontsize=12)

    for i in range(train_num):
        axs[i].set_ylim(min_y, max_y)
        if i != (train_num-1):
            axs[i].set_xticklabels([])
            axs[i].set_xlabel('')
        else:
            axs[i].set_xlabel('Seconds')

    #plt.savefig('POST -- Spike_Timing_OT -- Cell: ' + str(cell_num+1) + '.png')
    #plt.show()

    return 0

def get_stats(data, cell_indicies):
    start_col = cell_indicies[0]
    end_col = cell_indicies[1]
    train_num = end_col - start_col

    all_spike_nums = []
    all_mean_freq = []
    all_CV = []
    all_intervals = []

    for train_id in range(train_num):
        train_data = get_train(data, train_id, cell_indicies)

        spike_num = len(train_data)
        all_spike_nums.append(spike_num)

        intervals = np.diff(train_data)
        [all_intervals.append(interval) for interval in intervals]
        CV = np.std(intervals) / np.mean(intervals)
        all_CV.append(CV)

        mean_inst_freq = np.mean((1 / intervals))  # add this to graph
        all_mean_freq.append(mean_inst_freq)

    mean_freq = np.mean(np.array(all_mean_freq))
    mean_CV = np.mean(all_CV)

    interval_array = np.array(all_intervals)

    return interval_array, mean_freq, mean_CV, all_mean_freq, all_CV

def cell_interval_histogram(interval_array, cell_num):

    plt.hist(interval_array, bins=45, color='blue')

    plt.title('Posterior Interval Histogram for Cell ' + str(cell_num+1))
    plt.xlabel('Inter-Spike Intervals (seconds)')
    plt.ylabel('Frequency')

    plt.ylim(bottom=-5)

    plt.tight_layout
    plt.show()

    return 0

def all_cells_histogram(all_cell_interval):
    blues = ['#4D4DFF', '#4545E6', '#3D3DCC', '#3535B3', '#2D2D99', '#252580', '#1D1D66', '#15154D', '#0D0D33', '#05051A']

    bin_num = 30
    for i, interval_array in enumerate(all_cell_interval):
        plt.hist(interval_array, alpha=0.5, bins=bin_num, color=blues[i], label=f'Cell {i+1}')

    plt.title(f'Posterior Interval Histogram for all Recorded Cells \n {bin_num} bins')
    plt.xlabel('Inter-Spike Intervals (seconds)')
    plt.ylabel('Log Frequency')
    plt.legend()
    plt.yscale('log')
    plt.ylim(bottom=-1)

    plt.tight_layout

    #plt.savefig('POST -- Log frequency histogram of all cells on same plot.png')
    plt.show()

    return 0


if __name__ == '__main__':
    data = pd.read_csv('data/posterior_data.csv')
    cell_indices = [(0, 9), (11, 20), (22, 30), (32, 38), (40, 44), (46, 52), (54, 61), (63, 71)]

    "--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"
    all_cells = []
    for cell_num in range(8):
        interval_array = get_stats(data, cell_indices[cell_num])
        #cell_interval_histogram(interval_array, cell_num)
        all_cells.append(interval_array)

    all_cells_histogram(all_cells)
