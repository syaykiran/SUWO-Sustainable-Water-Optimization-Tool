import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
#
# # Set font style
# plt.rcParams["font.size"] = 10
# plt.rcParams["font.family"] = "Palatino Linotype"
#
# # Colors
# colors = ["#0D95D0", "#E72F52", "#27AE60", "#F39C12", "#8E44AD", "#C0392B"]
# faded_colors = ["#AACBE9", "#F4ADA8", "#ABEBC6", "#F7DC6F", "#D7BDE2", "#E6B0AA"]
#
# # Metric labels
# metrics = ["Hydropower Gen (GWh)", "Irrigation Ratio", "Eco Dev Index"]
#
# # Data: Hydropower Generation, Irrigation Supply Ratio, Ecological Deviation Index
# scenarios = {
#     "NSGA-II Balanced": [1836, 0.96, 0.13],
#     "NSGA-II Energy": [1909, 0.87, 0.14],
#     "NSGA-II Irrigation": [1749, 1.00, 0.08],
#     "NSGA-II Eco": [1795, 0.92, 0.07],
#     "NSGA-III Balanced": [1846, 0.96, 0.11],
#     "NSGA-III Energy": [1887, 0.89, 0.13],
#     "NSGA-III Irrigation": [1794, 0.99, 0.08],
#     "NSGA-III Eco": [1781, 0.93, 0.08],
#     "Normal Year": [1839, 0.99, 0.12],
#     "Wet Year": [2196, 1.09, 0.49],
#     "Dry Year": [1272, 0.86, 0.14],
#     "Class A (Natural)": [1882, 0.90, 0.08],
#     "Class B (Slightly Mod.)": [1846, 0.96, 0.11],
#     "Class C (Moderately Mod.)": [1793, 1.00, 0.49],
#     "Class D (Largely Mod.)": [1786, 1.04, 1.39],
#     "Class E (Seriously Mod.)": [1788, 1.05, 2.97],
#     "Class F (Critically Mod.)": [1763, 1.06, 5.46]
# }
#
# # Normalize data
# data_array = np.array(list(scenarios.values()))
# normalized_values = []
#
# for i in range(len(metrics)):
#     values = data_array[:, i]
#     if i == 0:  # Hydropower (Higher is better)
#         norm_values = values / np.max(values)
#     else:  # Lower is better for Ecological Deviation Index
#         norm_values = np.min(values) / values
#     normalized_values.append(norm_values)
#
# # Convert back to list format for plotting
# scores = np.array(normalized_values).T
#
# # --- Radar Chart Function ---
# def radar_chart_polygon_overall(scores, metrics, scenario_labels, colors, faded_colors):
#     def radar_factory(num_vars):
#         theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
#
#         class RadarAxes(PolarAxes):
#             name = 'radar'
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
#     outer_polygon = np.ones(len(metrics)) * 1.2
#     ax.plot(theta, outer_polygon, color='#B5BABB', linewidth=5, linestyle='-', zorder=1)
#
#     # Set radar chart border styling
#     ax.spines['polar'].set_visible(True)
#     ax.spines['polar'].set_edgecolor("#C0C6C8")
#
#     # Plot each scenario
#     for i, (label, color, faded_color) in enumerate(zip(scenario_labels, colors, faded_colors)):
#         if i >= len(scores):  # Avoid extra colors if not enough data
#             break
#         stats = scores[i].tolist()
#         ax.plot(theta, stats, color=color, linewidth=2,
#                 marker='o', markersize=6, markerfacecolor=color,
#                 markeredgecolor="white", alpha=0.9, label=label)
#         ax.fill(theta, stats, facecolor=faded_color, alpha=0.5, label='_nolegend_')
#
#     ax.set_varlabels(metrics)
#     ax.set_ylim(0, 1.2)
#     ax.set_yticklabels([])
#     ax.legend(loc="upper right", frameon=False, fontsize=8)
#     plt.title("Scenario Comparison: Hydropower, Irrigation & Ecology")
#     ax.yaxis.grid(True, linestyle='dotted', linewidth=1, color='#A2A4A5')
#     plt.show()
#
# # Select top 6 scenarios for visualization (e.g., 2 per category)
# selected_scenarios = [
#     "NSGA-II Balanced", "NSGA-III Balanced", "Wet Year",
#     "Dry Year", "Class A (Natural)", "Class F (Critically Mod.)"
# ]
#
# selected_indices = [list(scenarios.keys()).index(s) for s in selected_scenarios]
# selected_scores = scores[selected_indices]
# selected_colors = [colors[i] for i in range(len(selected_scenarios))]
# selected_faded_colors = [faded_colors[i] for i in range(len(selected_scenarios))]
#
# # Plot radar chart
# radar_chart_polygon_overall(selected_scores, metrics, selected_scenarios, selected_colors, selected_faded_colors)


# ##import numpy as np
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
# color_1 = "#EFB743"  # energy
# color_2 = "#7DC462"  # irrigation
# color_3 = "#0D95D0"  # ecology
# faded_color_1 = "#F7DEB0"  # Faded blue for NSGA-II area
# faded_color_2 = "#C9E4BC"  # Faded red for NSGA-III area
# faded_color_3 = "#AACBE9"  # Faded red for NSGA-III area
# # Data
# categories = [
#     "NSGA-II Balanced",
#     "NSGA-III Balanced",
#     "NSGA-II Energy",
#     "NSGA-III Energy",
#     "NSGA-II Irrigation",
#     "NSGA-III Irrigation",
#     "NSGA-II Eco",
#     "NSGA-III Eco",
#     "Dry Year",
#     "Normal Year",
#     "Wet Year",
#     "F: Critically Mod.",
#     "E: Seriously Mod.",
#     "D: Largely Mod.",
#     "C: Mod. Mod.",
#     "B: Slightly Mod.",
#     "A: Natural",
# ]
#
# hydropower = [1836, 1846, 1909, 1887, 1749, 1794, 1795, 1781, 1272, 1839, 2196, 1763, 1788, 1786, 1793, 1846, 1882]
# irrigation = [0.96, 0.96, 0.87, 0.89, 1.00, 0.99, 0.92, 0.93, 0.86, 0.99, 1.09, 1.06, 1.05, 1.04, 1.00, 0.96, 0.90]
# eco_index = [0.13, 0.11, 0.14, 0.13, 0.08, 0.08, 0.07, 0.08, 0.14, 0.12, 0.49, 5.46, 2.97, 1.39, 0.49, 0.11, 0.08]
#
# import pandas as pd
#
# # Normalize data
# max_hydro, max_irrig = max(hydropower), max(irrigation)
# min_eco = min(eco_index)
#
# hydro_norm = np.array(hydropower) / max_hydro
# irrig_norm = np.array(irrigation) / max_irrig
# eco_norm = min_eco / np.array(eco_index)  # Lower is better, so invert
#
# # Combined normalized data
# normalized_data = np.array([hydro_norm, irrig_norm, eco_norm]).T
#
# df = pd.DataFrame(
#     normalized_data,
#     columns=["Hydropower_norm", "Irrigation_norm", "Eco_norm"],
#     index=categories
# )
#
# # CSV olarak kaydet
# df.to_csv("normalized_data.csv", index_label="Category")
#
# # --- Radar Chart Function ---
# def radar_chart(data, categories):
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
#     N = len(categories)
#     theta = radar_factory(N)
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'), dpi=200)
#     ax.set_facecolor("#E8EBEC")
#
#     # Plot Data
#     for i, (color, faded_color) in enumerate(zip([color_1, color_2, color_3], [faded_color_1, faded_color_2, faded_color_3])):
#         ax.plot(theta, data[:, i], color=color, linewidth=1.5, marker='o', markersize=7, markerfacecolor=color,
#                 markeredgecolor="white", alpha=0.9, label=f"Metric {i+1}")
#         ax.fill(theta, data[:, i], facecolor=faded_color, alpha=0.5, label='_nolegend_', zorder=1)
#
#     ax.set_varlabels(categories)
#     ax.set_ylim(0, 1.2)
#     ax.set_yticklabels([])
#     ax.legend(loc="upper right", frameon=False)
#     plt.title("Scenario Comparison - Hydropower, Irrigation, and Ecological Deviation")
#     ax.yaxis.grid(True, linestyle='dotted', linewidth=1, color='#A2A4A5')
#     plt.show()
#
# # Plot radar chart
# radar_chart(normalized_data, categories)

##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import pandas as pd

# --- Font ve renkler ---
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"

color_1 = "#EFB743"  # enerji
color_2 = "#7DC462"  # sulama
color_3 = "#0D95D0"  # ekoloji
faded_color_1 = "#F7DEB0"
faded_color_2 = "#C9E4BC"
faded_color_3 = "#AACBE9"

# --- Veriler ---
categories = [
    "NSGA-II Balanced","NSGA-III Balanced","NSGA-II Energy","NSGA-III Energy",
    "NSGA-II Irrigation","NSGA-III Irrigation","NSGA-II Eco","NSGA-III Eco",
    "Dry Year","Normal Year","Wet Year",
    "F: Critically Mod.","E: Seriously Mod.","D: Largely Mod.","C: Mod. Mod.",
    "B: Slightly Mod.","A: Natural"
]

# MW cinsinden kurulu güç veya yıllık üretim (GWh) tahmini
hydropower_MW = [1836, 1846, 1909, 1887, 1749, 1794, 1795, 1781, 1272, 1839, 2196, 1763, 1788, 1786, 1793, 1846, 1882]

# Sulama oranları
irrigation = [0.96, 0.96, 0.87, 0.89, 1.00, 0.99, 0.92, 0.93, 0.86, 0.99, 1.09, 1.06, 1.05, 1.04, 1.00, 0.96, 0.90]

# Ekosistem sapmaları
eco_index = [0.13, 0.11, 0.14, 0.13, 0.08, 0.08, 0.07, 0.08, 0.14, 0.12, 0.49, 5.46, 2.97, 1.39, 0.49, 0.11, 0.08]

# --- Normalize Veriler ---

# Enerji: MW -> GWh (yıl boyunca 100% kapasite varsayımı)
hours_per_year = 365*24
total_installed_power_MW = 70.6 + 278.4 + 97.44 + 160 + 37.9  # MW
practical_capacity_factor = 0.35  # pratikte kullanılabilir oran
max_energy = total_installed_power_MW * hours_per_year * practical_capacity_factor /1000 # GWh tahmini max
firm_energy_min = 1110  # GWh min 203.066+371.982+186.456+257.996+90.475=1,109.975GWh

# Kurulu güç değerlerini normalize et
# Burada hydropower_MW değerlerini GWh olarak kabul edelim (yaklaşık)
hydro_norm = [(val - firm_energy_min) / (max_energy - firm_energy_min) for val in hydropower_MW]
hydro_norm = np.clip(hydro_norm, 0, 1)

# Sulama normalize: 0.75–1 aralığı, 1’i geçenleri 1 yap
irrig_norm = [min(max((val - 0.75)/(1 - 0.75),0),1) for val in irrigation]

# Ekosistem normalize: 0 en iyi, 7.5 en kötü
eco_max = 7
eco_norm = [1 - min(val/eco_max, 1) for val in eco_index]  # ters çevir, 1 en iyi

# --- Normalized DataFrame ---
normalized_data = np.array([hydro_norm, irrig_norm, eco_norm]).T
df = pd.DataFrame(normalized_data, columns=["Hydropower_norm", "Irrigation_norm", "Eco_norm"], index=categories)
df.to_csv("normalized_data.csv", index_label="Category")
print("✅ normalized_data.csv oluşturuldu")

# --- Radar Chart Function ---
def radar_chart(data, categories):
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

    N = len(categories)
    theta = radar_factory(N)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'), dpi=200)
    ax.set_facecolor("#E8EBEC")

    # Plot Data
    for i, (color, faded_color) in enumerate(zip([color_1, color_2, color_3], [faded_color_1, faded_color_2, faded_color_3])):
        ax.plot(theta, data[:, i], color=color, linewidth=1.5, marker='o', markersize=7,
                markerfacecolor=color, markeredgecolor="white", alpha=0.9, label=f"Metric {i+1}")
        ax.fill(theta, data[:, i], facecolor=faded_color, alpha=0.5, label='_nolegend_')

    ax.set_varlabels(categories)
    ax.set_ylim(0, 1.2)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", frameon=False)
    plt.title("Scenario Comparison - Hydropower, Irrigation, and Ecological Deviation")
    ax.yaxis.grid(True, linestyle='dotted', linewidth=1, color='#A2A4A5')
    plt.show()

# --- Plot Radar Chart ---
radar_chart(normalized_data, categories)
