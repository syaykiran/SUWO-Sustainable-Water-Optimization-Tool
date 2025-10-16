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
from lib import *
# Set print options
np.set_printoptions(formatter={"float": lambda x: "{: .2f}".format(x)}, linewidth=150)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"
#
#
# #
# # Define scenario names
scenario_code_1, sce_label_1 = "SC04", "NSGA2"  # "EXP_SIM1", "NSGA2"
scenario_code_2, sce_label_2 = "SC05", "NSGA3"  # "EXP_NSGA3", "NSGA3"
# # #
# # # # Load the X and F data using scenario names and convert to numpy arrays
# # X1 = pd.read_csv(f"NEW_HIST_FILES\ALL_RES_X_{scenario_code_1}.csv").to_numpy()
# # F1 = pd.read_csv(f"NEW_HIST_FILES\ALL_RES_F_{scenario_code_1}.csv").to_numpy()
# # X2 = pd.read_csv(f"NEW_HIST_FILES\ALL_RES_X_{scenario_code_2}.csv").to_numpy()
# # F2 = pd.read_csv(f"NEW_HIST_FILES\ALL_RES_F_{scenario_code_2}.csv").to_numpy()
# #
# # # # ## Load historical data
# hist_F1 = []
# hist_F2 = []
# # #
# # # # Load total execution times, n_gen, and n_pop using scenario names
# tot_time_1 = import_value_hist(f"total_execution_time_{scenario_code_1}")
# tot_time_2 = import_value_hist(f"total_execution_time_{scenario_code_2}")
# n_gen_1 = import_value_hist(f"n_gen_{scenario_code_1}")
# n_gen_2 = import_value_hist(f"n_gen_{scenario_code_2}")
# n_pop_1 = import_value_hist(f"pop_size_{scenario_code_1}")
# n_pop_2 = import_value_hist(f"pop_size_{scenario_code_2}")
# # # import_value_hist
# # Check if n_gen and n_pop are the same for both scenarios
# if n_gen_1 != n_gen_2:
#     print(
#         f"Warning: n_gen values are different for the two scenarios ({n_gen_1} vs. {n_gen_2}). "
#         "This might lead to inconsistencies in the comparison."
#     )
# else:
#     n_gen = n_gen_1
#
# if n_pop_1 != n_pop_2:
#     print(
#         f"Warning: n_pop values are different for the two scenarios ({n_pop_1} vs. {n_pop_2}). "
#         "This might lead to inconsistencies in the comparison."
#     )
# else:
#     n_pop = n_pop_1
#
# n_gen =1000
# n_pop =1000
# n_evals = np.arange(n_gen, (n_gen * n_pop) + 1, n_gen)  # Calculate cumulative n_evals
#
# # ## Export
# hist_F1 = [pd.read_csv(f"NEW_HIST_FILES\hist_F_{scenario_code_1}_1_gen_{gen_num}.csv").values for gen_num in range(n_gen)]
# # hist_F2 = [pd.read_csv(f"NEW_HIST_FILES\hist_F_{scenario_code_2}_1_gen_{gen_num}.csv").values for gen_num in range(n_gen)]
# # # CALISTI
# #
# #
# all_data = []
# # Append all generations into one DataFrame
# for i, gen in enumerate(hist_F1):
#     df = pd.DataFrame(gen)
#     df["Generation"] = i  # Add generation info
#     all_data.append(df)
# # Save everything into one big CSV file
# pd.concat(all_data).to_csv(f"{scenario_code_1}_hist_F1_all.csv", index=False)
#
# # all_data = []
# # # Append all generations into one DataFrame
# # for i, gen in enumerate(hist_F2):
# #     df = pd.DataFrame(gen)
# #     df["Generation"] = i  # Add generation info
# #     all_data.append(df)
# # # Save everything into one big CSV file
# # pd.concat(all_data).to_csv(f"{scenario_code_2}_hist_F2_all.csv", index=False)
#
# df = pd.read_csv(f"{scenario_code_1}_hist_F1_all.csv")
# hist_F1 = [df[df["Generation"] == i].drop(columns=["Generation"]).values for i in range(1000)]

# df = pd.read_csv(f"{scenario_code_2}_hist_F2_all.csv")
# hist_F2 = [df[df["Generation"] == i].drop(columns=["Generation"]).values for i in range(1000)]


##export_data(n_evals,  "n_evals")
# export_data_hist(hist_F1, "hist_F1")
# export_data(F1[:, 1:],  "F1")
# export_data(hist_F2, "hist_F2")
# export_data(F2[:, 1:],  "F2")

## Load historical data from exported CSV files
# hist_F1 = get_data_hist("EXP_hist_F1")
# hist_F2 = get_data("EXP_hist_F2")

# hv1 = get_data("EXP_hv1")
# hv2 = get_data("EXP_hv2")

# F1 = get_data("EXP_F1") # CALISTI
# F2 = get_data("EXP_F2") # CALISTI
# n_evals = get_data("EXP_n_evals").flatten() # CALISTI



#
#
X1 = get_data("EXP_SIM1_RES_X_1")
F1 = get_data("EXP_SIM1_RES_F_1")
X2 = get_data("EXP_NSGA3_RES_X_1")
F2 = get_data("EXP_NSGA3_RES_F_1")

XA = get_data("EXP_RES_X_NS2-NATA_1")
FA = get_data("EXP_RES_F_NS2-NATA_1")
XC = get_data("EXP_RES_X_NS2-NATC_1")
FC = get_data("EXP_RES_F_NS2-NATC_1")
XD = get_data("EXP_RES_X_NS2-NATD_1")
FD = get_data("EXP_RES_F_NS2-NATD_1")
XE = get_data("EXP_RES_X_NS2-NATE_1")
FE = get_data("EXP_RES_F_NS2-NATE_1")
XF = get_data("EXP_RES_X_NS2-NATF_1")
FF = get_data("EXP_RES_F_NS2-NATF_1")

## Performance indicators
# ndsgr1, upm1, spm1, mct1 = calc_performance_indicators(hist_F1, n_pop, F1, tot_time_1, n_evals)
# ndsgr2, upm2, spm2, mct2 = calc_performance_indicators(hist_F2, n_pop, F2, tot_time_2, n_evals)
# print(f"Performance Indicators for {scenario_code_1}:")
# print(f"  NDSGR: {ndsgr1:.4f}")
# print(f"  UPM: {upm1:.4f}")
# print(f"  SPM: {spm1:.4f}")
# print(f"  MCT: {mct1:.4f}")
# print(f"\nPerformance Indicators for {scenario_code_2}:")
# print(f"  NDSGR: {ndsgr2:.4f}")
# print(f"  UPM: {upm2:.4f}")
# print(f"  SPM: {spm2:.4f}")
# print(f"  MCT: {mct2:.4f}")
## two algorithms Compare Plot
#
# plot_hypervolume_compare(Hypervolume, n_evals, sce_label_1, F1, hist_F1, sce_label_2, F2, hist_F2)
# plot_hypervolume_compare_inputhv(n_evals, "NSGA-II", "NSGA-III", normalize=False)# hazır hv varsa kullan


# plot_inverted_generational_distance_plus_compare(IGDPlus, n_evals, F1, hist_F_averaged, F2, hist_F2_averaged)
# plot_inverted_generational_distance_plus_compare_inputigdplus(n_evals, "NSGA-II", "NSGA-III")

# plot_hypervolume_compare_inputhv_range(n_evals, "NSGA-II", "NSGA-III", normalize=False)
# plot_hypervolume_compare_inputhv_range_turkish(n_evals, "NSGA-II", "NSGA-III", normalize=False)


# plot_inverted_generational_distance_compare(IGD, n_evals, F1, hist_F1, F2, hist_F2)
# plot_igdplus_compare_inputigdplus_range(n_evals, "NSGA-II", "NSGA-III", normalize=False)

## one algorithm Plot
# plot_hypervolume(Hypervolume, n_evals, F, hist_F)
# plot_inverted_generational_distance(IGD, n_evals, F, hist_F)
# plot_inverted_generational_distance_plus(IGDPlus, n_evals, F, hist_F)
##
# min_values, max_values = get_min_max_objectives(F1)
# filename = "PSEUDO_WEIGHTS_OUT.csv"
# RES_EVA, OUTFLOW, GEN_EN_GWH, IRG_WATER = get_results_from_pseudo_weights(X1, F1)

# get_balanced_energy_production
SCE_BALANCED_GEN_ENR_GWH = get_balanced_energy_production(X1, F1)
SCE_MAXENERG_GEN_ENR_GWH = get_maxenrgy_energy_production(X1, F1)
titles = get_data_string("CH_NAMES")
months = get_data_string("CH_MONTHS")
# gemplot_generated_energy(SCE_BALANCED_GEN_ENR_GWH, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months)

## get_compared_energy_production

# Define your weight sets
weight_sets = [
        (np.array([[0.33, 0.33, 0.33]]), "Balanced Allocation", "purple"),
        (np.array([[0.8, 0.1, 0.1]]), "Maximizing Energy", "orange"),
        (np.array([[0.1, 0.8, 0.1]]), "Sustainable Irrigation", "#008080"),
        (np.array([[0.1, 0.1, 0.8]]), "Eco-Centric Approach", "#00BFFF"),
    ]

# Get results for each scenario
scenarios = []
for weights, label, color in weight_sets:
    generated_energy_gwh = get_generated_energy_for_scenario(X1, F1, weights)
    scenarios.append((generated_energy_gwh, label, color))

# Get the other data you need for plotting
titles = get_data_string("CH_NAMES")

CH_NAMES_TURKISH2 = [
    "Yukarı Sakarya Alt Havzası",
    "Porsuk Alt Havzası",
    "Ankara Alt Havzası",
    "Gürsöğüt Barajı",
    "Kargı Barajı",
    "Kirmir Alt Havzası",
    "Sarıyar Barajı",
    "Gökçekaya Barajı",
    "Yenice Barajı",
    "Orta Sakarya Alt Havzası",
    "Göksu Alt Havzası",
    "Orta Sakarya Pamukova",
    "Aşağı Sakarya Alt Havzası",
    "Toplam"
]
#
# months = get_data_string("CH_MONTHS")
#
# gemplot_generated_energy_obs_line(SCE_BALANCED_GEN_ENR_GWH, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months)

# gemplot_generated_energy_scenarios_line(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months)
# gemplot_generated_energy_scenarios_line_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, CH_NAMES_TURKISH2, months)
# gemplot_generated_energy_scenarios_single_plots_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, CH_NAMES_TURKISH2, months)

# gemplot_generated_energy_scenarios(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months)

# plot_total_energy(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, months)
# plot_total_energy_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, months)

# # plot_yearly_total_energy_with_observation(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH)
# # plot_yearly_total_energy_without_observation(scenarios)
# plot_yearly_total_energy_without_observation_turkish(scenarios)

## line_gemplot_env_management_flow
SCE_BALANCED_ECO_A = get_balanced_ecology(XA, FA)
SCE_BALANCED_ECO_B = get_balanced_ecology(X1, F1)
SCE_BALANCED_ECO_C = get_balanced_ecology(XC, FC)
SCE_BALANCED_ECO_D = get_balanced_ecology(XD, FD)
SCE_BALANCED_ECO_E = get_balanced_ecology(XE, FE)
SCE_BALANCED_ECO_F = get_balanced_ecology(XF, FF)
titles = get_data_string("CH_NAMES")[:-1]
months = get_data_string("CH_MONTHS")
NAME_LIST = np.array(CH_NAMES_TURKISH2)
I = [node - 1 for node in IRG_NODES]
# Example usage
SIM_OUTPUT_LIST = [SCE_BALANCED_ECO_A[I], SCE_BALANCED_ECO_B[I], SCE_BALANCED_ECO_C[I],
                   SCE_BALANCED_ECO_D[I], SCE_BALANCED_ECO_E[I], SCE_BALANCED_ECO_F[I]]
selected_node_index = 4


# IRG_NODES = [10, 11, 12, 13]  # Adjust indices based on IRG_NODES [1, 2, 3, 6]

# line_gemplot_env_management_flow_single(SIM_OUTPUT_LIST, OBS_OUTFLOW[I], NAME_LIST[I], selected_node_index)
line_gemplot_env_management_flow_single_turkish(SIM_OUTPUT_LIST, OBS_OUTFLOW[I], NAME_LIST[I], selected_node_index)

# line_gemplot_env_management_flow(SIM_OUTPUT_LIST, OBS_OUTFLOW[I], NAME_LIST[I])
# line_gemplot_env_management_flow_turkish(SIM_OUTPUT_LIST, OBS_OUTFLOW[I], NAME_LIST[I])

# best_results = analyze_pseudo_weights_worst_best(X1, F1)  # Get the best_results list

# filename = "EXP_PSEWEI_SCE_NSG3.csv"
# best_results = analyze_pseudo_weights(X2, F2)  # Get the best_results list 3D PARETO REGARENK
# best_results = analyze_pseudo_weights_coordinate_plot(X2, F2)  # Get the best_results list  pARETO REGARENK

# analyze_pseudo_weights_animated(X1, F1, output_gif_name="pseudo_weights_analysis_animated_x1.gif", num_frames=120, rotation_speed=2)
# ##
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Load the data into a dictionary
# data = {
#     'Scenario': ['Reference', 'Balanced', 'Maximizing Energy', 'Sustainable Irrigation', 'Eco-Centric Approach'],
#     'Hydropower Generation (10^8 kWh)': [1601, 1836, 1909, 1749, 1795],
#     'Irrigation Water Supply (10^6 m^3)': [708, 637, 581, 666, 612],
#     'Ecological Deviation': [0.07, 0.13, 0.14, 0.08, 0.07]
# }
#
# # Convert the dictionary into a DataFrame
# df = pd.DataFrame(data)
#
# # Normalize each objective
# for col in ['Hydropower Generation (10^8 kWh)', 'Irrigation Water Supply (10^6 m^3)', 'Ecological Deviation']:
#     min_val = df[col].min()
#     max_val = df[col].max()
#     df[col] = (df[col] - min_val) / (max_val - min_val)
#
# # Melt the DataFrame to long format
# df_melt = df.melt(id_vars='Scenario', value_vars=['Hydropower Generation (10^8 kWh)', 'Irrigation Water Supply (10^6 m^3)', 'Ecological Deviation'])
#
# # Create the radar chart
# sns.set_theme(style="whitegrid")
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, polar=True)
#
# # Define the categories and angles for the radar chart
# categories = df_melt['variable'].unique()
# angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
# angles += angles[:1]
#
# # Plot each scenario
# for scenario in df['Scenario'].unique():
#     values = df_melt[df_melt['Scenario'] == scenario]['value'].tolist()
#     values += values[:1]
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=scenario)
#     ax.fill(angles, values, alpha=0.1)
#
# # Set the labels for the radar chart
# ax.set_thetagrids([angle * 180/3.14159 for angle in angles[:-1]], labels=categories)
# ax.set_title('Radar Chart of Scenarios (Normalized)', fontsize=14)
# ax.grid(True)
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()
#
# # filename = "COMP_OUT.csv"
# # RES_EVA, OUTFLOW, GEN_EN_GWH, IRG_WATER = get_results_from_compromise(X1, F1, filename=filename)

## Multi-Criteria Decision Making - High Trade-off Points
# filename = "EXP_COMP_PROG_SCE_NS2-NATF.csv"
# best_results = compromise_programming(X1, F1, filename=filename)  # Get the best_results list
#
# HighTradeoffPoints._do = corrected_do
#
# # Assuming 'nF' is your normalized Pareto front data
# dm = HighTradeoffPoints()
# I = dm(nF)  # Use nF here as well
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(nF[:, 0], nF[:, 1], nF[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="All Points")
# ax.scatter(nF[I, 0], nF[I, 1], nF[I, 2], alpha=0.6, facecolor="red", edgecolor='red', label="High Trade-off Points")
#
# ax.set_xlabel('Objective 1')
# ax.set_ylabel('Objective 2')
# ax.set_zlabel('Objective 3')
# ax.legend()
# plt.show()
##
# Example Usage
# hv_1 = get_data("EXP_hv1")  # NSGA-II
# hv_2 = get_data("EXP_hv2")  # NSGA-III
#
# hvis_results = compute_hvis(hv_1, hv_2)
# print(hvis_results)

##

# F1_name="NSGA-II"
# F2_name="NSGA-III"
#
# plot_igdplus_compare_inputigdplus_range(n_evals, F1_name, F2_name, normalize=False)