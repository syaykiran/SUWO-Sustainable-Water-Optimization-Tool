import numpy as np
from lib import plot_boxplot_outflow_scatter, get_data_string, plot_boxplot_outflow, plot_scatter_collect, plot_boxplot_collect, plot_boxplot, simulation_model, read_penalty_coef
from load import *
import csv

NAME_LIST = np.array(CH_NAMES)

# Get simulation results
total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
SIM_OUTFLOW = []

X1 = get_data("EXP_SIM1_RES_X_1")  # Load data from EXP_SIM1_RES_X_1

# Assuming X1 is an ndarray of size (1000, 156) and T = 12
for idx, (K_1, K_2, K_3, K_SC) in enumerate(coef_values, start=1):
    # Iterate over rows of X1 using slicing
    for row_idx in range(X1.shape[0]):
        row = X1[row_idx, :]  # Get the current row
        DEC_VARS = np.array([float(val) for val in row]).reshape(13, 12)  # Reshape to (13, 12)
        RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
        ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
            POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, \
            PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE = RESULT

        SIM_OUTFLOW.append(OUTFLOW)

SIM_OUTFLOW = np.array(SIM_OUTFLOW)


# plot_boxplot_outflow(SIM_OUTFLOW, OBS_OUTFLOW, NAT_EMC, NAME_LIST, debug_mode=True, debug_idx=12)
plot_boxplot_outflow_scatter(SIM_OUTFLOW, OBS_OUTFLOW, NAT_EMC, NAME_LIST, debug_mode=True, debug_idx=12)
