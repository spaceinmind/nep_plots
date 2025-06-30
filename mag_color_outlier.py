import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

# Constants
XMIN, XMAX = -50.0, 50.0
YMIN, YMAX = XMIN, XMAX
NBIN = 30
CONTOURS = [0.9973, 0.9545, 0.6827]

# Helper function for contour level calculation
def objective(limit, target, counts):
    return counts[counts > limit].sum() - target

# Load data
data = pd.read_csv('/fred/oz002/users/sho/nep/tosimon.csv', delimiter=',', dtype='object')
data = data.astype({
    'HSC_R': float,
    'Maidanak_R': float,
    'MEGACAM_r': float
})

# Derived columns
data['HSC_R-Maidanak_R'] = data['HSC_R'] - data['Maidanak_R']
data['HSC_R-MEGACAM_r'] = data['HSC_R'] - data['MEGACAM_r']

# Apply clipping
mask0 = data['HSC_R'].between(XMIN, XMAX) & data['HSC_R-Maidanak_R'].between(YMIN, YMAX)
data0 = data[mask0].reset_index(drop=True)

mask1 = data['HSC_R'].between(XMIN, XMAX) & data['HSC_R-MEGACAM_r'].between(YMIN, YMAX)
data1 = data[mask1].reset_index(drop=True)

# Statistics
mean1, std1 = data0['HSC_R-Maidanak_R'].mean(), data0['HSC_R-Maidanak_R'].std()
mean2, std2 = data1['HSC_R-MEGACAM_r'].mean(), data1['HSC_R-MEGACAM_r'].std()

# Plot setup
fig, axarr = plt.subplots(2, 2, sharey=True)

# Adjust subplot positions
adjustments = [
    ((0, 0), 0.1, 1.4),
    ((1, 0), 0.1, 1.4),
    ((0, 1), 0.2, 0.45),
    ((1, 1), 0.2, 0.45)
]
for (i, j), x0_offset, width_scale in adjustments:
    box = axarr[i, j].get_position()
    axarr[i, j].set_position([box.x0 + x0_offset, box.y0, box.width * width_scale, box.height])

# Scatter plots
axarr[0, 0].scatter(data0['HSC_R'], data0['HSC_R-Maidanak_R'], s=0.7, alpha=0.9, c='k', marker='.')
axarr[1, 0].scatter(data1['HSC_R'], data1['HSC_R-MEGACAM_r'], s=0.7, alpha=0.9, c='k', marker='.')

# Horizontal lines
for ax, mean, std in zip([axarr[0, 0], axarr[1, 0]], [mean1, mean2], [std1, std2]):
    ax.axhline(mean, color='k', alpha=0.5, linestyle='-')
    ax.axhline(mean + 3 * std, color='k', alpha=0.5, linestyle='--')
    ax.axhline(mean - 3 * std, color='k', alpha=0.5, linestyle='--')

# Labels
axarr[0, 0].set_ylabel('HSC_R - Maidanak_R', fontsize=14, labelpad=13)
axarr[1, 0].set_ylabel('HSC_R - MEGACAM_r', fontsize=14, labelpad=13)
axarr[1, 0].set_xlabel('HSC_R', fontsize=18, labelpad=13)

# Contour calculation function
def plot_contours(ax, xdata, ydata, levels):
    counts, xbins, ybins = np.histogram2d(xdata, ydata, bins=NBIN, density=True)
    norm = counts.sum()
    targets = [norm * c for c in levels]
    lvl_values = [
        scipy.optimize.bisect(objective, counts.min(), counts.max(), args=(t, counts)) for t in targets
    ] + [counts.max()]
    x = np.linspace(xdata.min(), xdata.max(), NBIN)
    y = np.linspace(ydata.min(), ydata.max(), NBIN)
    ax.contour(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        linewidths=1,
        colors='red',
        linestyles='solid',
        levels=lvl_values
    )

# Contour plots
plot_contours(axarr[0, 0], data0['HSC_R'], data0['HSC_R-Maidanak_R'], CONTOURS)
plot_contours(axarr[1, 0], data1['HSC_R'], data1['HSC_R-MEGACAM_r'], CONTOURS)

# Histograms
axarr[0, 1].hist(data0['HSC_R-Maidanak_R'], bins=100, histtype='stepfilled', orientation='horizontal', color='gray')
axarr[1, 1].hist(data1['HSC_R-MEGACAM_r'], bins=100, histtype='stepfilled', orientation='horizontal', color='gray')

plt.show()
