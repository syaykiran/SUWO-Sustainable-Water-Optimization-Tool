import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from lib import *
from matplotlib.path import Path
import matplotlib.patches as patches

# --- Load data from CSV files --- (Assuming you have this function)
RES_F1 = get_data("EXP_SIM1_RES_F_1")  # Load data for NSGA-II
RES_F2 = get_data("EXP_NSGA3_RES_F_1")  # Load data for NSGA-III

# Apply Seaborn style
sns.set(style="whitegrid")

# Set font parameters after setting Seaborn style
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"
sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Palatino Linotype', 'font.size': '10'})

# Define colors
color_1 = "#0D95D0"  # Teal 008080
color_2 = "#E72F52"  # Red f44336

color_1_hues = ['#0480B6', '#0D95D0', '#58A5D7']  # Light blue to dark blue
color_2_hues = ['#D22946', '#E72F52', '#EB636A']  # Light red to dark red

# Data values for NSGA-II and NSGA-III
data = {
    'NSGA-II': {
        'HV': 3.5,
        'UPM': 0.0115,
        'SPM': 1.139,
        'NDSGR': 0.9262,
        'MCT': 6.093
    },
    'NSGA-III': {
        'HV': 2.75,
        'UPM': 0.0072,
        'SPM': 0.785,
        'NDSGR': 0.508,
        'MCT': 5.797
    }
}

# Indicator names
indicators = ['HV', 'UPM', 'SPM', 'NDSGR', 'MCT']

# Indicator types (whether each indicator should be maximized or minimized)
indicator_types = ['maximize', 'minimize', 'maximize', 'maximize', 'minimize']

# Example usage with your data
# compare_models_polar(data, indicators, indicator_types)

# --- Plot the Pareto fronts in separate plots ---
plot_3d_pareto_compare(RES_F1, RES_F2, color_1_hues, color_2_hues)
# plot_2d_pareto_compare(RES_F1, RES_F2, color_1_hues, color_2_hues)
# parallel_coordinates_plot(RES_F1, RES_F2, color_1_hues=color_1_hues, color_2_hues=color_2_hues)
# parallel_coordinates_plot(RES_F1, RES_F2, color_1_hues=['#0480B6', '#0D95D0', '#58A5D7'],
#                              color_2_hues=['#D22946', '#E72F52', '#EB636A'])
