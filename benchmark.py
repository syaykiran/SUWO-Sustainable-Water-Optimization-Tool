import numpy as np
from lib import gemplot_irrigation_box, get_balanced_irrigation, read_penalty_coef, simulation_model_inflow_scenario
from load import *

# IRG BOXPLOT GRAFIKLERI

# IRG_NODES = [10, 11, 12, 13]  # Adjust indices based on IRG_NODES [1, 2, 3, 6]
I = [node - 1 for node in IRG_NODES]
NAME_LIST = np.array(CH_NAMES)

# Initialize lists to store simulation results for each model
SIM_IRG_WATER_1 = []

# Read penalty coefficients for each simulation
total_iter_1, iter_name_1, coef_values_1 = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
#
# Loop through each iteration
for idx in range(total_iter_1):
    # Get coefficient values for the first simulation for the current iteration
    K_1_1, K_2_1, K_3_1, K_SC_1 = coef_values_1[idx]

    # Read data for the first simulation
    df = pd.read_csv(f"EXP_SIM1_RES_X_{idx + 1}.csv").iloc[:, 1:]

    # Assuming all simulations have the same number of rows (iterations)
    for index in range(len(df)):
        # Get DEC_VARS for the first simulation
        dec_vars = np.array([float(val) for val in df.iloc[index, :]]).reshape(13, -1)

        # Run the first simulation and store results
        result_1 = simulation_model_inflow_scenario(dec_vars, K_1_1, K_2_1, K_3_1, K_SC_1, inflow_scenario="NAT_INF_NOR")
        (ENR_OBJ_1, IRG_OBJ_1, ECO_OBJ_1, ENG_TOT_1, IRG_TOT_1,
         TOT_ECO_DEV_1, AVE_REG_RATIO_1, SPILLWAY_1, GEN_WATER_1, IRG_WATER_1,
         GEN_EN_GWH_1, POWER_MW_1, POW_MW_AVE_1, STORAGES_1, RES_EVA_1,
         RES_ELEV_1, OUTFLOW_1, PEN_MIN_SQTOT_1, PEN_MAX_SQTOT_1,
         PEN_END_SQTOT_1, PEN_ALL_TOTAL_1, STO_DIF_1, IRG_DEF_SQTOT_1,
         PEN_SCALE_1) = result_1

        # Store results in a 3D array structure
        SIM_IRG_WATER_1.append(IRG_WATER_1[I, :])  # 2 nodes, 12 months

# Convert lists to 3D NumPy arrays
SIM_IRG_WATER_1 = np.array(SIM_IRG_WATER_1)  # Shape: (num_iterations, 2, 12)
X1 = get_data("EXP_SIM1_RES_X_1")
F1 = get_data("EXP_SIM1_RES_F_1")

Balanced = np.array([[0.33, 0.33, 0.33]])  # "Balanced", "purple"
MaximizingEnergy = np.array([[0.8, 0.1, 0.1]])  # "Maximizing Energy", "orange"
SustainableIrrigation = np.array([[0.1, 0.8, 0.1]])  # "Sustainable Irrigation", "#008080"
EcoCentric = np.array([[0.1, 0.1, 0.8]])  # "Eco-Centric Approach", "#00BFFF"

SCE_BALANCED_TOT_IRG_WATER = np.array(get_balanced_irrigation(X1, F1, Balanced)[I, :])
SCE_MAXENRGY_TOT_IRG_WATER = np.array(get_balanced_irrigation(X1, F1, MaximizingEnergy)[I, :])
SCE_SUSIRGAT_TOT_IRG_WATER = np.array(get_balanced_irrigation(X1, F1, SustainableIrrigation)[I, :])
SCE_ECOCENTR_TOT_IRG_WATER = np.array(get_balanced_irrigation(X1, F1, EcoCentric)[I, :])
## Plot the irrigation results
gemplot_irrigation_box(SIM_IRG_WATER_1, IR_DEMAND[I], NAME_LIST[I], MONTHS_NO, LOWER_BOUND_CSV[I], UPPER_BOUND_CSV[I],
                       SCE_BALANCED_TOT_IRG_WATER, SCE_MAXENRGY_TOT_IRG_WATER, SCE_SUSIRGAT_TOT_IRG_WATER, SCE_ECOCENTR_TOT_IRG_WATER)
