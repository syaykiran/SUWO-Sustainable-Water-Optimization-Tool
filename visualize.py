import matplotlib.pyplot as plt
import numpy as np
from lib import (export_data, simulation_model, plot_pareto_mesh, plot_obj_total, plot_objectives, plot_energy_sector,
                 plot_reservoir_volume, plot_pareto_2obj, plot_reservoir_levels, plot_reservoir_net_evaporation,
                 plot_spillway, plot_pareto_3d, plot_pareto_2d, get_data_string, get_data, get_dec_var_best,
                 get_dec_var_cust, read_penalty_coef, import_value, export_data, get_x_initials)
from load import *
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

# Set numpy print options and matplotlib default font
np.set_printoptions(formatter={"float": lambda x: "{: .2f}".format(x)}, linewidth=150)
total_iter, iter_name, coef_values = read_penalty_coef("EXP_NSGA3_PEN_COEF")
RES_F = get_data("EXP_RES_F_NSG3_CLMTAVOF3_1")
RES_F_NUM = RES_F.shape[0]
# DEC_VARS_BEST = get_x_initials(OBS_DEMAND)  # for CALIBRATION
# DEC_VARS_BEST = np.reshape(OBS_DEMAND, (13, 12)) # for CALIBRATION
DEC_VARS_BEST = get_dec_var_cust("EXP_RES_X_NSG3_CLMTAVOF3_1", 0)
K_1, K_2, K_3, K_SC = coef_values[0]

RESULT = simulation_model(DEC_VARS_BEST, K_1, K_2, K_3, K_SC)
ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, \
PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE = RESULT


# plot_pareto_2obj(RES_F)
# plot_reservoir_volume(STORAGES, OBS_STO, S_MAX, S_0, S_MIN, VERSION="VERSION", FOOT_NOTE="FOOT_NOTE")
# plot_pareto_3d(RES_F)
# plot_pareto_2d(RES_F)
# plot_objectives(RES_COUNT, GEN_WATER, IRG_WATER, EN_DEMAND, IR_DEMAND, OUTFLOW, OBS_NAT_FLOW, UPPER_BOUND_CSV, LOWER_BOUND_CSV, OBS_EN_MIN, OBS_EN_MAX)
# plot_obj_total(RES_COUNT, GEN_WATER, EN_DEMAND, IRG_WATER, IR_DEMAND, OUTFLOW, OBS_OUTFLOW, OBS_NAT_FLOW)
# plot_energy_sector(GEN_EN_GWH, OBS_EN_GWH, GEN_WATER, EN_DEMAND, POWER_MW, OBS_EN_FIRM, OBS_EN_FIRM_MW, OBS_EN_FIRM_GWH)
plot_reservoir_levels(OBS_ELEV, RES_ELEV, S_MAX_LEVEL, S_MIN_LEVEL)
plot_reservoir_net_evaporation(OBS_EVA, RES_EVA)
plot_spillway(SPILLWAY, SPILLAGE_CAP)
plt.show()

