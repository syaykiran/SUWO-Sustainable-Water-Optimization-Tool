import numpy as np
from lib import get_data_string, plot_boxplot_outflow, plot_scatter_collect, plot_boxplot_collect, plot_boxplot, simulation_model, read_penalty_coef
from load import *
import csv

total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
E = [node - 1 for node in ENG_NODES]
NAME_LIST = np.array(CH_NAMES)

SIM_STORAGES = []
SIM_RES_ELEV = []
SIM_OUTFLOW = []

for idx, (K_1, K_2, K_3, K_SC) in enumerate(coef_values, start=1):
    df = pd.read_csv(f"EXP_RES_X_{idx}.csv")
    df = df.iloc[:, 1:]

    for index, row in df.iterrows():
        DEC_VARS = np.array([float(val) for val in row[:]]).reshape(13, T)
        RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
        ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
            POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, \
            PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE = RESULT

        SIM_STORAGES.append(STORAGES[E, :])
        SIM_RES_ELEV.append(RES_ELEV[E, :])
        SIM_OUTFLOW.append(OUTFLOW)

SIM_STORAGES = np.array(SIM_STORAGES)
SIM_RES_ELEV = np.array(SIM_RES_ELEV)
SIM_OUTFLOW = np.array(SIM_OUTFLOW)

print("Progress completed.")

# plot_boxplot(SIM_RES_ELEV, OBS_ELEV[E], NAME_LIST[E], S_MIN_LEVEL[E], S_MAX_LEVEL[E], debug_mode=False, debug_idx=0)
# plot_scatter_collect(SIM_STORAGES, OBS_STO[E], NAME_LIST[E], MONTHS_NO, S_MIN[E], S_MAX[E], S_0[E])
plot_boxplot_outflow(SIM_OUTFLOW, OBS_OUTFLOW, NAT_EMC, NAME_LIST, debug_mode=True, debug_idx=12)
