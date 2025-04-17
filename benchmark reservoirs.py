import numpy as np
from lib import gemplot_scatter_collect, get_data_string, plot_boxplot_outflow, plot_scatter_collect, plot_boxplot_collect, \
    plot_boxplot, simulation_model_inflow_scenario, read_penalty_coef
from load import *
import csv
E = [node - 1 for node in ENG_NODES]  # Adjust indices based on ENG_NODES
I = [1, 2]  # Adjust indices based on ENG_NODES

NAME_LIST = np.array(CH_NAMES)

# Initialize lists to store simulation results for each model
SIM_STORAGES_1 = []
SIM_STORAGES_2 = []
SIM_STORAGES_3 = []

SIM_RES_ELEV_1 = []
SIM_RES_ELEV_2 = []
SIM_RES_ELEV_3 = []

SIM_OUTFLOW_1 = []
SIM_OUTFLOW_2 = []
SIM_OUTFLOW_3 = []

SIM_IRG_WATER_1 = []
SIM_IRG_WATER_2 = []
SIM_IRG_WATER_3 = []

# Read penalty coefficients for each simulation
total_iter_1, iter_name_1, coef_values_1 = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
total_iter_2, iter_name_2, coef_values_2 = read_penalty_coef("SIM_WET_NSG2_PEN_COEF")
total_iter_3, iter_name_3, coef_values_3 = read_penalty_coef("SIM_DRY_NSG2_PEN_COEF")

# Loop through each iteration
for idx in range(total_iter_1):
    # Get coefficient values for each simulation for the current iteration
    K_1_1, K_2_1, K_3_1, K_SC_1 = coef_values_1[idx]
    K_1_2, K_2_2, K_3_2, K_SC_2 = coef_values_2[idx]
    K_1_3, K_2_3, K_3_3, K_SC_3 = coef_values_3[idx]

    # Read data for all simulations using a loop
    dfs = [pd.read_csv(f"EXP_SIM{i}_RES_X_{idx + 1}.csv").iloc[:, 1:] for i in range(1, 4)]

    # Assuming all simulations have the same number of rows (iterations)
    for index in range(len(dfs[0])):  # Use the first DataFrame for length
        # Get DEC_VARS for each simulation
        dec_vars = [np.array([float(val) for val in df.iloc[index, :]]).reshape(13, -1) for df in dfs]

        # Run simulations and store results for each simulation
        result_1 = simulation_model_inflow_scenario(dec_vars[0], K_1_1, K_2_1, K_3_1, K_SC_1, inflow_scenario="NAT_INF_NOR")  # Simulation 1
        (ENR_OBJ_1, IRG_OBJ_1, ECO_OBJ_1, ENG_TOT_1, IRG_TOT_1,
         TOT_ECO_DEV_1, AVE_REG_RATIO_1, SPILLWAY_1, GEN_WATER_1, IRG_WATER_1,
         GEN_EN_GWH_1, POWER_MW_1, POW_MW_AVE_1, STORAGES_1, RES_EVA_1,
         RES_ELEV_1, OUTFLOW_1, PEN_MIN_SQTOT_1, PEN_MAX_SQTOT_1,
         PEN_END_SQTOT_1, PEN_ALL_TOTAL_1, STO_DIF_1, IRG_DEF_SQTOT_1,
         PEN_SCALE_1) = result_1

        # Store results in a 3D array structure
        SIM_STORAGES_1.append(STORAGES_1[E, :])  # 5 nodes, 12 months
        SIM_RES_ELEV_1.append(RES_ELEV_1[E, :])  # 5 nodes, 12 months
        SIM_IRG_WATER_1.append(IRG_WATER_1[I, :])  # 5 nodes, 12 months
        SIM_OUTFLOW_1.append(OUTFLOW_1)  # Assuming it's a single value or 1D array

        result_2 = simulation_model_inflow_scenario(dec_vars[1], K_1_2, K_2_2, K_3_2, K_SC_2, inflow_scenario="NAT_INF_WET")  # Simulation 2
        (ENR_OBJ_2, IRG_OBJ_2, ECO_OBJ_2, ENG_TOT_2, IRG_TOT_2,
         TOT_ECO_DEV_2, AVE_REG_RATIO_2, SPILLWAY_2, GEN_WATER_2, IRG_WATER_2,
         GEN_EN_GWH_2, POWER_MW_2, POW_MW_AVE_2, STORAGES_2, RES_EVA_2,
         RES_ELEV_2, OUTFLOW_2, PEN_MIN_SQTOT_2, PEN_MAX_SQTOT_2,
         PEN_END_SQTOT_2, PEN_ALL_TOTAL_2, STO_DIF_2, IRG_DEF_SQTOT_2,
         PEN_SCALE_2) = result_2

        SIM_STORAGES_2.append(STORAGES_2[E, :])
        SIM_RES_ELEV_2.append(RES_ELEV_2[E, :])
        SIM_IRG_WATER_2.append(IRG_WATER_2[I, :])  # 5 nodes, 12 months
        SIM_OUTFLOW_2.append(OUTFLOW_2)

        result_3 = simulation_model_inflow_scenario(dec_vars[2], K_1_3, K_2_3, K_3_3, K_SC_3, inflow_scenario="NAT_INF_DRY")  # Simulation 3
        (ENR_OBJ_3, IRG_OBJ_3, ECO_OBJ_3, ENG_TOT_3, IRG_TOT_3,
         TOT_ECO_DEV_3, AVE_REG_RATIO_3, SPILLWAY_3, GEN_WATER_3, IRG_WATER_3,
         GEN_EN_GWH_3, POWER_MW_3, POW_MW_AVE_3, STORAGES_3, RES_EVA_3,
         RES_ELEV_3, OUTFLOW_3, PEN_MIN_SQTOT_3, PEN_MAX_SQTOT_3,
         PEN_END_SQTOT_3, PEN_ALL_TOTAL_3, STO_DIF_3, IRG_DEF_SQTOT_3,
         PEN_SCALE_3) = result_3

        SIM_STORAGES_3.append(STORAGES_3[E, :])
        SIM_RES_ELEV_3.append(RES_ELEV_3[E, :])
        SIM_IRG_WATER_3.append(IRG_WATER_3[I, :])  # 5 nodes, 12 months
        SIM_OUTFLOW_3.append(OUTFLOW_3)

# Convert lists to 3D NumPy arrays
SIM_STORAGES_1 = np.array(SIM_STORAGES_1)  # Shape: (num_iterations, 5, 12)
SIM_STORAGES_2 = np.array(SIM_STORAGES_2)  # Shape: (num_iterations, 5, 12)
SIM_STORAGES_3 = np.array(SIM_STORAGES_3)  # Shape: (num_iterations, 5, 12)

SIM_RES_ELEV_1 = np.array(SIM_RES_ELEV_1)  # Shape: (num_iterations, 5, 12)
SIM_RES_ELEV_2 = np.array(SIM_RES_ELEV_2)  # Shape: (num_iterations, 5, 12)
SIM_RES_ELEV_3 = np.array(SIM_RES_ELEV_3)  # Shape: (num_iterations, 5, 12)

# Convert lists to 3D NumPy arrays
SIM_IRG_WATER_1 = np.array(SIM_IRG_WATER_1)  # Shape: (num_iterations, 5, 12)
SIM_IRG_WATER_2 = np.array(SIM_IRG_WATER_2)  # Shape: (num_iterations, 5, 12)
SIM_IRG_WATER_3 = np.array(SIM_IRG_WATER_3)  # Shape: (num_iterations, 5, 12)

# Convert OUTFLOW to a suitable shape
SIM_OUTFLOW_1 = np.array(SIM_OUTFLOW_1)  # Adjust if needed to match desired shape
SIM_OUTFLOW_2 = np.array(SIM_OUTFLOW_2)  # Adjust if needed to match desired shape
SIM_OUTFLOW_3 = np.array(SIM_OUTFLOW_3)  # Adjust if needed to match desired shape

# Create a list of 3D NumPy arrays for each simulation
SIM_STORAGES = [SIM_STORAGES_1, SIM_STORAGES_2, SIM_STORAGES_3]
SIM_IRG_WATERS = [SIM_IRG_WATER_1, SIM_IRG_WATER_2, SIM_IRG_WATER_3]

print("Progress completed.")
## Boxplot with Scatter for Reservoir
# plot_boxplot(SIM_RES_ELEV, OBS_ELEV[E], NAME_LIST[E], S_MIN_LEVEL[E], S_MAX_LEVEL[E], debug_mode=False, debug_idx=0)
# plot_scatter_collect(SIM_STORAGES, OBS_STO[E], NAME_LIST[E], MONTHS_NO, S_MIN[E], S_MAX[E], S_0[E])
gemplot_scatter_collect(SIM_STORAGES, OBS_STO[E], NAME_LIST[E], MONTHS_NO, S_MIN[E], S_MAX[E], S_0[E])
# gemplot_scatter_collect(SIM_IRG_WATERS, OBS_STO[I], NAME_LIST[I], MONTHS_NO, S_MIN[I], S_MAX[I], S_0[I])

# plot_boxplot_outflow(SIM_OUTFLOW_1, OBS_OUTFLOW, NAT_EMC, NAME_LIST, debug_mode=True, debug_idx=12)
