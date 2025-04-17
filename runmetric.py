import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.running_metric import RunningMetricAnimation
from pymoo.mcdm.compromise_programming import CompromiseProgramming
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.decomposition.asf import ASF
import csv
import os
from load import *
from lib import export_running_igd, plot_running_igd, compute_running_igd
# Set print options
np.set_printoptions(formatter={"float": lambda x: "{: .2f}".format(x)}, linewidth=150)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"

# Define scenario names
scenario_code_1, sce_label_1 = "NSG3_SEED4", "NSGA"
scenario_code_2, sce_label_2 = "NSG3_SEED5", "NSGA"
# "SC04", "NSGA2" / "SC05", "NSGA3"

# Define export/import functions
def xexport_data(data, filename):
    """Exports a numpy array or list of numpy arrays to a CSV file."""
    if isinstance(data, np.ndarray):
        # Single numpy array
        np.savetxt(f"{filename}.csv", data, delimiter=",")
    elif isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
        # List of numbers
        with open(f"{filename}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
    else:
        print(f"Unsupported data type for export: {type(data)}")

def ximport_data(filename):
    """Imports data from a CSV file."""
    if os.path.exists(filename):
        try:
            return np.loadtxt(filename, delimiter=",")
        except ValueError:
            # Try reading as a single row of numbers
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    try:
                        return [float(val) for val in row]
                    except ValueError:
                        return None #Or handle as needed
        except Exception as e:
            print(f"Error importing {filename}: {e}")
            return None
    else:
        print(f"File not found: {filename}")
        return None

def xexport_data_hist(hist_F, filename):
    """Exports a list of numpy arrays (history data) to a single CSV file."""
    all_data = []
    for i, arr in enumerate(hist_F):
        df = pd.DataFrame(arr)
        df["Generation"] = i  # Add generation info
        all_data.append(df)
    pd.concat(all_data).to_csv(f"{filename}.csv", index=False)

def ximport_data_hist(filename):
    """Imports history data from a single CSV file."""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        max_gen = df["Generation"].max()
        hist_F = [df[df["Generation"] == i].drop(columns=["Generation"]).values for i in range(max_gen + 1)]
        return hist_F
    else:
        print(f"File not found: {filename}")
        return None


# n_gen = 1000
# n_pop = 1000

# # Example loading of hist_F1, replace with your actual data loading
# hist_F1 = []
# for gen_num in range(n_gen):
#     filename = f"NEW_HIST_FILES/hist_F_{scenario_code_1}_1_gen_{gen_num}.csv"
#     if os.path.exists(filename):
#         hist_F1.append(pd.read_csv(filename).values)
#     else:
#         print(f"Warning: File {filename} not found.")
#
# hist_F2 = []
# for gen_num in range(n_gen):
#     filename = f"NEW_HIST_FILES/hist_F_{scenario_code_2}_1_gen_{gen_num}.csv"
#     if os.path.exists(filename):
#         hist_F2.append(pd.read_csv(filename).values)
#     else:
#         print(f"Warning: File {filename} not found.")

# Calculate cumulative n_evals
# n_evals = np.arange(n_gen, (n_gen * n_pop) + 1, n_gen)

# Export data
# xexport_data(n_evals, "n_evals")
# xexport_data_hist(hist_F1, "EXP_hist_F1") # export to one file
# xexport_data_hist(hist_F2, "EXP_hist_F2") # export to one file
##
# n_evals = ximport_data("n_evals.csv")
# hist_F1 = ximport_data_hist("hist_F1_all.csv")
# hist_F2 = ximport_data_hist("hist_F2_all.csv")
##
# scenario_code_1_filename = f"{scenario_code_1}_running_igd.csv"
# scenario_code_2_filename = f"{scenario_code_2}_running_igd.csv"
#
# # Bulk execution for multiple models
# models = [(scenario_code_1_filename, hist_F1), (scenario_code_2_filename, hist_F2)]
#
# for save_path, hist_F in models:
#     export_running_igd(n_evals, hist_F, window_size=50, save_path=save_path)
# plot_running_igd(n_evals, hist_F1, "NSGA-II", window_size=50)
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

color_1 = "#0D95D0"  # BL for NSGA-II
color_2 = "#E72F52"  # RD for NSGA-III
faded_color_1 = "#AACBE9"
faded_color_2 = "#F4ADA8"

# Function to load and aggregate IGD values
def load_and_aggregate_igd(pattern):
    files = glob.glob(pattern)
    all_data = []

    for file in files:
        df = pd.read_csv(file)
        all_data.append(df["running_igd"].values)

    all_data = np.array(all_data)  # Shape: (num_seeds, num_generations)

    mean_igd = np.mean(all_data, axis=0)
    min_igd = np.min(all_data, axis=0)
    max_igd = np.max(all_data, axis=0)

    return df["n_evals"].values, mean_igd, min_igd, max_igd, files


# Function to plot comparison of NSGA2 vs NSGA3 with min-max shading
def plot_nsga_comparison():
    n_evals, mean_igd_nsga2, min_igd_nsga2, max_igd_nsga2, _ = load_and_aggregate_igd("NSG2_SEED*_running_igd.csv")
    _, mean_igd_nsga3, min_igd_nsga3, max_igd_nsga3, _ = load_and_aggregate_igd("NSG3_SEED*_running_igd.csv")

    plt.figure(figsize=(7, 5), dpi=200)

    # NSGA2
    plt.plot(n_evals, mean_igd_nsga2, color=color_1, lw=1.2, alpha=0.9, label="NSGA-II")
    plt.fill_between(n_evals, min_igd_nsga2, max_igd_nsga2, color=faded_color_1, alpha=0.5)

    # NSGA3
    plt.plot(n_evals, mean_igd_nsga3, color=color_2, lw=1.2, alpha=0.9, label="NSGA-III")
    plt.fill_between(n_evals, min_igd_nsga3, max_igd_nsga3, color=faded_color_2, alpha=0.5)

    plt.yscale("log")
    plt.xlabel("Function Evaluations")
    plt.ylabel(r"$\emptyset$ - Running IGD")
    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    # plt.title("Comparison of NSGA2 vs NSGA3 (∅ - Running IGD)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.show()


def plot_all_individual_runs():
    nsga2_files = glob.glob("NSG2_SEED*_running_igd.csv")
    nsga3_files = glob.glob("NSG3_SEED*_running_igd.csv")

    plt.figure(figsize=(10, 6))

    # Plot NSGA2 runs
    for file in nsga2_files:
        df = pd.read_csv(file)
        plt.plot(df["n_evals"].to_numpy(), df["running_igd"].to_numpy(),
                 label=f"NSGA2 {file.split('_')[2]}", linestyle='dashed')

    # Plot NSGA3 runs
    for file in nsga3_files:
        df = pd.read_csv(file)
        plt.plot(df["n_evals"].to_numpy(), df["running_igd"].to_numpy(),
                 label=f"NSGA3 {file.split('_')[2]}")

    plt.yscale("log")
    plt.xlabel("Function Evaluations")
    plt.ylabel("∅ - Running IGD")
    plt.title("Individual Runs for NSGA-II & NSGA-III")
    plt.legend(ncol=2, fontsize="small")  # Multi-column legend for clarity
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Example usage:
plot_nsga_comparison()  # Plot aggregated NSGA2 vs NSGA3
# plot_all_individual_runs()  # Plot all 10 runs individually
