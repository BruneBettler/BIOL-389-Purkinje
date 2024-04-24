# Author: Brune Bettler
# Date: April 7th 2024
import matplotlib
import pandas as pd
from anterior_analysis import get_stats
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.integrate import quad
from scipy.stats import skew
from scipy.stats import lognorm, kstest
import seaborn as sns

post_data = pd.read_csv('data/posterior_data.csv')
post_cell_indices = [(0, 9), (11, 20), (22, 30), (32, 38), (40, 44), (46, 52), (54, 61), (63, 71)]

ant_data = pd.read_csv('data/anterior_data.csv')
ant_cell_indices = [(0, 8), (10, 19), (21, 27), (29, 38), (40, 49), (51, 60), (62, 71), (73, 82), (84, 93), (95, 104)]

# grouped histogram
ant_cell_sep = []
for i, indices in enumerate(ant_cell_indices):
    ant_cell_sep.append(get_stats(ant_data, indices))

post_cell_sep = []
for i, indices in enumerate(post_cell_indices):
    post_cell_sep.append(get_stats(post_data, indices))


ant_cells = [item for sublist in ant_cell_sep for item in sublist]
post_cells = [item for sublist in post_cell_sep for item in sublist]

ant_cells_normalized = np.log(np.array(ant_cells)/len(ant_cells))
post_cells_normalized = np.log(np.array(post_cells)/len(post_cells))

ant_kde = gaussian_kde(ant_cells_normalized)
ant_x_grid = np.linspace(ant_cells_normalized.min(), ant_cells_normalized.max(), 2000)  # Adjust 1000 as needed for resolution
ant_kde_values = ant_kde(ant_x_grid)
ant_peaks, _ = find_peaks(ant_kde_values)
plt.plot(ant_x_grid, ant_kde_values, color='red', label='Anterior KDE')
plt.scatter(ant_x_grid[ant_peaks][4:7], ant_kde_values[ant_peaks][4:7], color='r', zorder=5)

post_kde = gaussian_kde(post_cells_normalized)
post_x_grid = np.linspace(post_cells_normalized.min(), post_cells_normalized.max(), 2000)  # Adjust 1000 as needed for resolution
post_kde_values = post_kde(post_x_grid)
post_peaks, _ = find_peaks(post_kde_values)
plt.plot(post_x_grid, post_kde_values, color='blue', label='Posterior KDE')
plt.scatter(post_x_grid[post_peaks][0:3], post_kde_values[post_peaks][0:3], color='blue', zorder=5)

plt.xlabel('Normalized Log Inter-Spike Interval (s)')
plt.legend()
plt.ylabel('Probability Density')
plt.title('Kernel Density Estimate Plot \n for Normalized Log Inter-Spike Intervals')
plt.savefig('KDE normalized plot.png')
plt.show()


# Define a function that computes the minimum of the two KDEs at any given point
def min_kdes(x):
    return min(ant_kde(x), post_kde(x))

# Compute the overlap coefficient by integrating the minimum of the KDEs
# Assuming the range of interest covers most of the data
x_min = min(min(ant_cells_normalized), min(post_cells_normalized))
x_max = max(max(ant_cells_normalized), max(post_cells_normalized))

overlap_area, _ = quad(min_kdes, x_min, x_max)

print(f'Overlap Coefficient: {overlap_area}')
print(skew(ant_cells_normalized))
print(skew(post_cells_normalized))


''' 
Create Histogram to compare all spikes anterior vs posterior 
'''

'''ant_spikes = pd.DataFrame(np.log(ant_cells_normalized), columns=['Spike Times'])
post_spikes = pd.DataFrame(np.log(post_cells_normalized), columns=['Spike Times'])

sns.histplot(kde=True, edgecolor=None, data=ant_spikes, color='red', alpha=0.3, x='Spike Times', bins=32, label="Anterior")
plt.fill_betweenx(y=[0, 6000], x1=min(np.log(ant_cells_normalized)), x2=max(np.log(ant_cells_normalized)), color='red', alpha=0.1)

sns.histplot(kde=True, edgecolor=None, data=post_spikes, color='blue', alpha=0.3, x='Spike Times', bins=32, label="Posterior")
plt.fill_betweenx(y=[0, 6000], x1=min(np.log(post_cells_normalized)), x2=max(np.log(post_cells_normalized)), color='blue', alpha=0.1)

plt.xlabel('Normalized Log Inter-Spike Interval (s)')
plt.legend()
plt.ylabel('Frequency of Occurance')
plt.title('Normalized Histogram of Inter-Spike Intervals')
'''
#plt.savefig('Normalized histogram ant vs post with lines.png')

#plt.show()

'''
# make two histograms on same plot: red ant, blue post
plt.hist(ant_cells, bins=32, density=True, color='red', alpha=0.3, label='Anterior Cells')
plt.hist(post_cells, bins=150, density=True, color='blue', alpha=0.3, label='Posterior Cells')
plt.yscale('log')

plt.title('Probability Density Function: Anterior vs Posterior Spike-Timing Intervals \n Fitted Log-Normal Distribution Curve')
plt.xlabel('Inter-Spike Intervals (seconds)')
plt.ylabel('Log Frequency')
plt.legend()

plt.tight_layout()

plt.show()'''

''' ------------------------------------------------------------- '''

