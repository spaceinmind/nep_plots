# North Ecliptic Pole (NEP) AKARI plotting scripts

This repository contains Python scripts to plot figures that are presented in [Ho et al. 2021, MNRAS](https://academic.oup.com/mnras/article/502/1/140/6122898?login=false). The visualizations include scatter plots, confidence contours, and histograms and filter transmission curves.

## Features

- Compare photometry from two datasets and plot the 2D histogram-based contours with confidence levels of the distribution
- WCS-FITS image mosaics with source overlays in different bands
- Plot filter transmission/response curves

## Usage and example plots
- Filter transmission/response curves of HSC
``` python 
python script/filter_response.py
```
<img src="plots/filter_response_example.png" alt="filter_response_example" width="800">  

- WCS-FITS image mosaics with source overlays in 7 bands
``` python script/
python script/fitsinput.py
```
<img src="plots/fits_input_example_plot.png" alt="fits_input_example_plot" width="800">

- Magnitude vs colors plots for HSC_R, Maidanak_R and MegaCam_r
``` python script/
python script/fitsinput.py
```
<img src="plots/mag_color_outlier_example_plot.png" alt="mag_color_outlier_example_plot" width="800">  

## Citation
Please cite [Ho et al. 2021, MNRAS](https://academic.oup.com/mnras/article/502/1/140/6122898?login=false) if you use the code in your paper.
