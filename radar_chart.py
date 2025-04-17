# import numpy as np
# from pathlib2 import Path
# PATHWAY = Path(".")
# def get_data(file_name):
#     csv_name_path = PATHWAY / (file_name + ".csv")
#     data = np.genfromtxt(csv_name_path, delimiter=",", skip_header=1)
#     # Check if the array has only one dimension
#     if data.ndim == 1:
#         # For one-dimensional array, remove the first element
#         data = data[1:].reshape(1, -1)
#     else:
#         # For two-dimensional array, skip the first column
#         data = data[:, 1:]
# 
#     return data
#
# def compute_ideal_hvis(n_evals, hv_1, hv_2, ideal_n_evals=1000):
#     """
#     Computes the Ideal HVIS for a given number of evaluations.
#
#     Args:
#         n_evals (np.ndarray): Function evaluations at each step.
#         hv_1 (np.ndarray): Hypervolume values for Algorithm 1.
#         hv_2 (np.ndarray): Hypervolume values for Algorithm 2.
#         ideal_n_evals (int): Number of evaluations for the ideal HVIS computation.
#
#     Returns:
#         float: The ideal HVIS score.
#     """
#
#     # Find the maximum hypervolume achieved by either algorithm
#     HV_max = max(np.max(hv_1), np.max(hv_2))
#
#     # Create an ideal HV curve where HV_max is reached instantly
#     ideal_hv_curve = np.full(ideal_n_evals, HV_max)
#
#     # Generate an evaluation range from 1 to ideal_n_evals
#     ideal_n_evals_range = np.linspace(1, ideal_n_evals, ideal_n_evals)
#
#     # Compute the ideal HVIS using numerical integration (Trapezoidal Rule)
#     ideal_hvis = np.trapz(ideal_hv_curve, ideal_n_evals_range)
#
#     return ideal_hvis
#
# # Example Usage
# n_evals = np.linspace(0, 1000000, num=100)  # Function evaluations
#
# hv_1 = get_data("EXP_hv1")  # NSGA-II
# hv_2 = get_data("EXP_hv2")  # NSGA-III
#
# ideal_hvis = compute_ideal_hvis(n_evals, hv_1, hv_2, ideal_n_evals=1000)
# print("Ideal HVIS for 1000 evaluations:", ideal_hvis)

##
# ## CALISMADI RUNNING
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pymoo.visualization.video.callback_video import AnimationCallback
# from pymoo.core.callback import Callback
# from pymoo.indicators.hv import Hypervolume
# from pymoo.indicators.igd import IGD
# from pymoo.termination.ftol import calc_delta_norm
# from pymoo.util.normalization import normalize
# from pymoo.util.sliding_window import SlidingWindow
#
# # Mock Algorithm class to mimic algorithm behavior for RunningMetricAnimation
# class MockAlgorithm:
#     def __init__(self, n_gen, hist_F1):
#         self.n_gen = n_gen  # Current generation
#         # Make sure to pass the right data from hist_F1 for this generation
#         self.opt = {"F": hist_F1[n_gen]["F"]}  # Current objective values (F)
#
# # RunningMetric callback for calculating the running metric
# class RunningMetric(Callback):
#     def __init__(self, period=None, indicator="igd") -> None:
#         super().__init__()
#         self.indicator = indicator
#         self.delta_ideal = None
#         self.delta_nadir = None
#         self.delta_f = None
#         self.history = SlidingWindow(period)
#
#     def update(self, algorithm):
#         history = self.history
#         F = algorithm.opt.get("F")
#         c_F, c_ideal, c_nadir = F, F.min(axis=0), F.max(axis=0)
#         history.append(dict(F=F, ideal=c_ideal, nadir=c_nadir))
#
#         # the current norm that should be used for normalization
#         norm = c_nadir - c_ideal
#         norm[norm < 1e-32] = 1.0
#
#         # normalize the current objective space values
#         c_N = normalize(c_F, c_ideal, c_nadir)
#
#         # normalize all previous generations with respect to current ideal and nadir
#         N = [normalize(e["F"], c_ideal, c_nadir) for e in history]
#
#         # calculate the delta difference for each previous ideal and nadir point to current
#         delta_ideal = [calc_delta_norm(history[k]["ideal"], history[k - 1]["ideal"], norm) for k in range(1, len(history))] + [0.0]
#         delta_nadir = [calc_delta_norm(history[k]["nadir"], history[k - 1]["nadir"], norm) for k in range(1, len(history))] + [0.0]
#
#         # now calculate the indicator from each previous one to the current
#         if self.indicator == "igd":
#             delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]
#         elif self.indicator == "hv":
#             hv = Hypervolume(ref_point=np.ones(c_F.shape[1]))
#             delta_f = [hv.do(N[k]) for k in range(len(N))]
#         else:
#             raise Exception("Unknown indicator.")
#
#         self.delta_ideal, self.delta_nadir, self.delta_f = delta_ideal, delta_nadir, delta_f
#
#
# # RunningMetricAnimation for visualization
# class RunningMetricAnimation(AnimationCallback):
#     def __init__(self, delta_gen, n_plots=4, key_press=True, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.running = RunningMetric()
#         self.delta_gen = delta_gen
#         self.key_press = key_press
#         self.data = SlidingWindow(n_plots)
#
#     def draw(self, data, ax):
#         for tau, x, f, v in data[:-1]:
#             ax.plot(x, f, label="t=%s" % tau, alpha=0.6, linewidth=3)
#
#         tau, x, f, v = data[-1]
#         ax.plot(x, f, label="t=%s (*)" % tau, alpha=0.9, linewidth=3)
#
#         for k in range(len(v)):
#             if v[k]:
#                 ax.plot([k + 1, k + 1], [0, f[k]], color="black", linewidth=0.5, alpha=0.5)
#                 ax.plot([k + 1], [f[k]], "o", color="black", alpha=0.5, markersize=2)
#
#         ax.set_yscale("symlog")
#         ax.legend()
#         ax.set_xlabel("Generation")
#         ax.set_ylabel("$\Delta \, f$", rotation=0)
#
#     def do(self, _, algorithm, force_plot=False, **kwargs):
#         running = self.running
#         running.update(algorithm)
#
#         tau = algorithm.n_gen
#
#         print(f"Generation {tau}: Checking data collection condition...")
#
#         # Only append data if the conditions are met
#         if (tau > 0 and tau % self.delta_gen == 0) or force_plot:
#             print(f"Appending data for generation {tau}.")
#             f = running.delta_f
#             x = np.arange(len(f)) + 1
#             v = [max(ideal, nadir) > 0.005 for ideal, nadir in zip(running.delta_ideal, running.delta_nadir)]
#             self.data.append((tau, x, f, v))
#         else:
#             print(f"Skipping data collection for generation {tau}. Condition not met.")
#
#         # Check if there is any data in self.data before attempting to plot
#         if len(self.data) > 0:
#             fig, ax = plt.subplots()
#             self.draw(self.data, ax)
#
#             if self.key_press:
#                 def press(event):
#                     if event.key == 'q':
#                         algorithm.termination.force_termination = True
#
#                 fig.canvas.mpl_connect('key_press_event', press)
#
#             plt.draw()
#             plt.waitforbuttonpress()
#             plt.close('all')
#         else:
#             print(f"No data collected for generation {tau}, skipping plotting.")
#
#
# # Function to run the Running Metric Animation using data
# def running_metric(hist_F1, delta_gen=50, n_plots=4):
#
#     # Initialize RunningMetricAnimation
#     running = RunningMetricAnimation(delta_gen=delta_gen, n_plots=n_plots, key_press=True, do_show=True)
#
#     # Loop through your data (hist_F1) and call running.do() for each generation
#     for gen_index, gen_data in enumerate(hist_F1):
#         # Create a mock algorithm object for this generation
#         algorithm = MockAlgorithm(n_gen=gen_index, hist_F1=hist_F1)
#
#         # Call the do() method with the mock algorithm and generation data
#         running.do(gen_data, algorithm)
#
#
# # Load your data from CSV file
# df = pd.read_csv("hist_F1_all.csv", header=None, names=["F1", "F2", "F3", "Generation"], low_memory=False)
#
# # Group by generation and extract the values for each generation
# generations = df.groupby("Generation")
#
# # Prepare the formatted history (hist_F1)
# hist_F1 = []
#
# for gen, group in generations:
#     F = group[["F1", "F2", "F3"]].values  # The objective values (F)
#     ideal = F.min(axis=0)  # Ideal (min) values for each objective
#     nadir = F.max(axis=0)  # Nadir (max) values for each objective
#
#     hist_F1.append({
#         "F": F,        # Objective values for this generation
#         "ideal": ideal,  # Ideal points for this generation
#         "nadir": nadir   # Nadir points for this generation
#     })
#
# # Run the metric animation
# running_metric(hist_F1)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# color_1 = "#0D95D0"  # Blue for NSGA-II
# color_2 = "#E72F52"  # Red for NSGA-III
#
# faded_color_1 = "#AACBE9"
# faded_color_2 = "#F4ADA8"
# # Original metric values
# metrics = ["HV", "HV (area)", "IDG+", "UPM", "SPM", "NDSGR", "MCT"]
# nsg2_values = [3.166, 817, 0.104, 0.012, 1.139, 0.926, 6.093]
# nsg3_values = [3.863, 742, 0.189, 0.007, 0.785, 0.508, 5.797]
#
# # Ideal Ranges (min, max)
# ideal_ranges = {
#     "HV": (2.684, 3.602),
#     "HV (area)": (742, 817),
#     "IDG+": (0.001, 0.189),  # Minimize
#     "UPM": (0, np.inf),  # Minimize (normalized differently)
#     "SPM": (0, np.inf),
#     "NDSGR": (0, 1),
#     "MCT": (0, np.inf)  # Minimize
# }
#
#
# # Normalization function
# def normalize(value, metric, ideal_ranges):
#     min_ideal, max_ideal = ideal_ranges[metric]
#
#     if metric in ["IDG+", "UPM", "MCT"]:  # Minimize
#         return 1 - (value - min_ideal) / (max_ideal - min_ideal) if max_ideal < np.inf else 1 / (1 + value)
#     else:  # Maximize
#         return (value - min_ideal) / (max_ideal - min_ideal) if max_ideal < np.inf else np.log1p(value)
#
#
# # Normalize all metrics
# nsg2_normalized = [normalize(val, metric, ideal_ranges) for val, metric in zip(nsg2_values, metrics)]
# nsg3_normalized = [normalize(val, metric, ideal_ranges) for val, metric in zip(nsg3_values, metrics)]
#
# # Convert to DataFrame
# df_normalized = pd.DataFrame({"Metric": metrics, "NSGA-II": nsg2_normalized, "NSGA-III": nsg3_normalized})
#
#
# # --- PLOT: Radar Chart ---
# def radar_chart(df):
#     labels = df["Metric"].values
#     stats_nsg2 = df["NSGA-II"].values
#     stats_nsg3 = df["NSGA-III"].values
#
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     stats_nsg2 = np.concatenate((stats_nsg2, [stats_nsg2[0]]))  # Close the plot
#     stats_nsg3 = np.concatenate((stats_nsg3, [stats_nsg3[0]]))
#     angles += angles[:1]
#
#     fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
#     ax.fill(angles, stats_nsg2, color=color_1, alpha=0.9, label="NSGA-II")
#     ax.fill(angles, stats_nsg3, color=color_2, alpha=0.9, label="NSGA-III")
#
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=12)
#     ax.legend(loc="upper right")
#     plt.title("Performance Comparison (Radar Chart)")
#     plt.show()
#
#
# # --- PLOT: Bar Chart ---
# def bar_chart(df):
#     df.plot(x="Metric", kind="bar", figsize=(10, 5), color=[color_1, color_2])
#     plt.title("Normalized Comparison of NSGA-II vs NSGA-III")
#     plt.ylabel("Normalized Score (0-1)")
#     plt.xticks(rotation=45)
#     plt.legend(title="Algorithm")
#     plt.show()
#

# Plot results
# radar_chart(df_normalized)
# bar_chart(df_normalized)
# ## first code
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import RegularPolygon
# from matplotlib.path import Path
# from matplotlib.projections import register_projection
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.spines import Spine
# from matplotlib.transforms import Affine2D
#
# # Set font style
# plt.rcParams["font.size"] = 10
# plt.rcParams["font.family"] = "Palatino Linotype"
#
# # Colors
# color_1 = "#0D95D0"  # Blue for NSGA-II
# color_2 = "#E72F52"  # Red for NSGA-III
# faded_color_1 = "#AACBE9"  # Faded blue for NSGA-II area
# faded_color_2 = "#F4ADA8"  # Faded red for NSGA-III area
#
# # Metric data for each model (NSGA-II and NSGA-III)
# nsg2_seeds = {
#     "seed1": [3.166, 817, 0.104, 0.012, 1.139, 0.926, 6.093],
#     "seed2": [3.123, 810, 0.111, 0.013, 1.080, 0.927, 7.116],
#     "seed3": [3.201, 801, 0.101, 0.007, 1.257, 0.920, 6.583],
#     "seed4": [3.189, 808, 0.108, 0.013, 1.236, 0.926, 6.494],
#     "seed5": [3.145, 805, 0.105, 0.007, 1.194, 0.919, 6.677]
# }
#
# nsg3_seeds = {
#     "seed1": [3.863, 742, 0.189, 0.007, 0.785, 0.508, 5.797],
#     "seed2": [3.811, 738, 0.181, 0.013, 0.761, 0.533, 6.833],
#     "seed3": [3.892, 745, 0.185, 0.007, 0.746, 0.503, 5.950],
#     "seed4": [3.845, 740, 0.183, 0.013, 0.822, 0.484, 6.103],
#     "seed5": [3.823, 735, 0.180, 0.007, 0.868, 0.478, 7.655]
# }
#
# metrics = ["HV", "HV (area)", "IDG+", "UPM", "SPM", "NDSGR", "MCT"]
# maximize_metrics = {"HV", "HV (area)", "NDSGR"}  # Higher is better
# minimize_metrics = {"IDG+", "UPM", "SPM", "MCT"}  # Lower is better
#
# # Convert data into structured format
# all_data = np.array(list(nsg2_seeds.values()) + list(nsg3_seeds.values()))
# metric_values = np.array(all_data).T  # Transpose to get metrics as rows
#
#
# # --- SCORING FUNCTION ---
# def compute_scores(metric_values, metrics, maximize_metrics):
#     scores = []
#     for i, metric in enumerate(metrics):
#         values = metric_values[i]
#         if metric in maximize_metrics:
#             normalized = values / np.max(values)  # Normalize by max value
#         else:
#             normalized = np.min(values) / values  # Normalize by min value (inverted)
#         scores.append(normalized)
#     return np.array(scores).T  # Transpose back for easy plotting
#
#
# # Compute normalized scores
# scores = compute_scores(metric_values, metrics, maximize_metrics)
#
#
# # --- PLOT: Radar Chart (Polygon) ---
# def radar_chart_polygon(scores, metrics, models, colors, faded_colors):
#     def radar_factory(num_vars, frame='polygon'):
#         theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
#
#         class RadarTransform(PolarAxes.PolarTransform):
#             def transform_path_non_affine(self, path):
#                 if path._interpolation_steps > 1:
#                     path = path.interpolated(num_vars)
#                 return Path(self.transform(path.vertices), path.codes)
#
#         class RadarAxes(PolarAxes):
#             name = 'radar'
#             PolarTransform = RadarTransform
#
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)
#                 self.set_theta_zero_location('N')
#
#             def fill(self, *args, closed=True, **kwargs):
#                 return super().fill(closed=closed, *args, **kwargs)
#
#             def plot(self, *args, **kwargs):
#                 lines = super().plot(*args, **kwargs)
#                 for line in lines:
#                     self._close_line(line)
#
#             def _close_line(self, line):
#                 x, y = line.get_data()
#                 if x[0] != x[-1]:
#                     x = np.append(x, x[0])
#                     y = np.append(y, y[0])
#                     line.set_data(x, y)
#
#             def set_varlabels(self, labels):
#                 self.set_thetagrids(np.degrees(theta), labels)
#
#             def _gen_axes_patch(self):
#                 return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
#
#             def _gen_axes_spines(self):
#                 spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
#                 spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
#                 return {'polar': spine}
#
#         register_projection(RadarAxes)
#         return theta
#
#     N = len(metrics)
#     theta = radar_factory(N)
#
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
#     ax.set_facecolor("#E8EBEC")
#
#     # Add outer polygon for better illustration
#     outer_polygon = np.ones(len(metrics)) * 1.2  # Slightly extends the max value
#     ax.plot(theta, outer_polygon, color='#B5BABB', linewidth=5, linestyle='-', zorder=1)
#     # ax.scatter(theta, outer_polygon, color="gray", marker='o', s=50, zorder=5)  # Markers at corners
#
#     # Set radar chart border styling
#     ax.spines['polar'].set_visible(True)
#     ax.spines['polar'].set_edgecolor("#C0C6C8")  # Gray border
#
#     # Plot models' data
#     for i, (model, color, faded_color) in enumerate(zip(models, colors, faded_colors)):
#         stats = scores[i].tolist()
#         ax.plot(theta, stats, color=color, linewidth=2,
#                 marker='o', markersize=7, markerfacecolor=color,
#                 markeredgecolor="white", alpha=0.9, label=model)
#         ax.fill(theta, stats, facecolor=faded_color, alpha=0.5, label='_nolegend_')
#
#     ax.set_varlabels(metrics)
#     ax.set_ylim(0, 1.2)  # Extend beyond 1 for better visualization
#     ax.set_yticklabels([])  # Remove axis labels but keep gridlines
#     ax.legend(loc="upper right", frameon=False)
#     plt.title("Model Comparison (Normalized Scores)")
#     ax.yaxis.grid(True, linestyle='dotted', linewidth=1, color='#A2A4A5')  # Dotted inner grid lines
#     plt.show()
#
#
# # Plot radar chart (polygon)
# models = ["NSGA-II (Avg)", "NSGA-III (Avg)"]
# colors = [color_1, color_2]
# faded_colors = [faded_color_1, faded_color_2]
# avg_scores = np.mean(scores.reshape(2, 5, -1), axis=1)  # Average across seeds
# radar_chart_polygon(avg_scores, metrics, models, colors, faded_colors)

##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Set font style
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"

# Colors
color_1 = "#0D95D0"  # Blue for NSG2
color_2 = "#E72F52"  # Red for NSG3
faded_color_1 = "#AACBE9"  # Faded blue for NSG2 area
faded_color_2 = "#F4ADA8"  # Faded red for NSG3 area

# Metric labels
metrics = [
    "HV AV", "HV FINAL", "IGD+ AV", "IGD+ FINAL", "RUNNING IGD AV",
    "RUNNING IGD FINAL", "UPM", "SPM", "NDSGR", "MCT"
]

metrics = [
    "HV AV", "HV FINAL", "IGD+ AV", "IGD+ FINAL", "RUNNING IGD AV",
    "RUNNING IGD FINAL", "UPM", "SPM", "NDSGR", "MCT"
]

# Average metric values for NSG2 and NSG3
nsg2_values = [3.341, 4.053, 0.155, 0.044, 0.184, 0.014, 0.010, 1.181, 0.924, 6.593]
nsg3_values = [2.860, 3.786, 0.180, 0.014, 0.192, 0.013, 0.009, 0.796, 0.501, 6.468]

# Metrics where higher values are better (maximize)
maximize_metrics = {"HV AV", "HV FINAL", "NDSGR", "SPM"}

# Metrics where lower values are better (minimize)
minimize_metrics = {"IGD+ AV", "IGD+ FINAL", "RUNNING IGD AV", "RUNNING IGD FINAL", "UPM", "SPM", "MCT"}

# Normalize data
all_values = np.array([nsg2_values, nsg3_values])
normalized_values = []

for i, metric in enumerate(metrics):
    values = all_values[:, i]  # Get values for both NSG2 and NSG3
    if metric in maximize_metrics:
        norm_values = values / np.max(values)  # Normalize by max value
    else:
        norm_values = np.min(values) / values  # Normalize by min value (inverted)
    normalized_values.append(norm_values)

# Convert back to list format for plotting
scores = np.array(normalized_values).T

# --- Radar Chart Function ---
def radar_chart_polygon(scores, metrics, models, colors, faded_colors):
    def radar_factory(num_vars, frame='polygon'):
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):
            def transform_path_non_affine(self, path):
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):
            name = 'radar'
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")

            def _gen_axes_spines(self):
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}

        register_projection(RadarAxes)
        return theta


    N = len(metrics)
    theta = radar_factory(N)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'), dpi=200)
    ax.set_facecolor("#E8EBEC")

    # Add outer polygon for better illustration
    outer_polygon = np.ones(len(metrics)) * 1.2  # Slightly extends the max value
    ax.plot(theta, outer_polygon, color='#B5BABB', linewidth=5, linestyle='-', zorder=1)

    # Set radar chart border styling
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_edgecolor("#C0C6C8")  # Gray border

    # Plot models' data
    for i, (model, color, faded_color) in enumerate(zip(models, colors, faded_colors)):
        stats = scores[i].tolist()
        ax.plot(theta, stats, color=color, linewidth=2,
                marker='o', markersize=7, markerfacecolor=color,
                markeredgecolor="white", alpha=0.9, label=model)
        ax.fill(theta, stats, facecolor=faded_color, alpha=0.5, label='_nolegend_')

    ax.set_varlabels(metrics)
    ax.set_ylim(0, 1.2)  # Extend beyond 1 for better visualization
    ax.set_yticklabels([])  # Remove axis labels but keep gridlines
    ax.legend(loc="upper right", frameon=False)
    plt.title("NSG2 vs NSG3 (Normalized Scores)")
    ax.yaxis.grid(True, linestyle='dotted', linewidth=1, color='#A2A4A5')  # Dotted inner grid lines
    plt.show()


# Plot radar chart
models = ["NSGA-II", "NSGA-III"]
colors = [color_1, color_2]
faded_colors = [faded_color_1, faded_color_2]

radar_chart_polygon(scores, metrics, models, colors, faded_colors)
