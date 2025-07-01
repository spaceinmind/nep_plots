import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np

# Set up figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Read data
data = pd.read_csv('HSC_g.csv', delimiter=',', dtype='float', header=None)
data1 = pd.read_csv('HSC_r.csv', delimiter=',', dtype='float', header=None)
data2 = pd.read_csv('HSC_i.csv', delimiter=',', dtype='float', header=None)
data3 = pd.read_csv('HSC_z.csv', delimiter=',', dtype='float', header=None)
data4 = pd.read_csv('HSC_y.csv', delimiter=',', dtype='float', header=None)

# List of datasets, labels, and colors
datasets = [
    (data, 'g', 'b'),
    (data1, 'r', 'g'),
    (data2, 'i', 'r'),
    (data3, 'z', 'm'),
    (data4, 'y', 'k')
]

text_kwargs = dict(fontsize=16, ha='center', va='bottom', alpha=0.7)

# Plot filters
for d, label, color in datasets:
    ax.fill(d[0], d[1], color=color, alpha=0.4, label=f'HSC-{label}')
    peak_wavelength = d[0][d[1].idxmax()]
    ax.text(peak_wavelength, 0.02, label, color=color, **text_kwargs)

# Configure plot
ax.set_xlim(3000, 11000)
ax.set_ylim(0, 1.1)
ax.set_title('HSC Filters', fontsize=18)
ax.set_xlabel('Wavelength (Å)', fontsize=14)
ax.set_ylabel('Normalized Transmission', fontsize=14)
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()
