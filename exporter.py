import matplotlib.pyplot as plt
import numpy as np
from lib import simulation_model, plot_pareto_mesh, plot_obj_total, plot_objectives, plot_energy_sector, \
    plot_reservoir_volume, \
    plot_reservoir_levels, plot_reservoir_net_evaporation, \
    plot_spillway, plot_pareto_3d, plot_pareto_2d, get_data_string, get_data, get_dec_var_best, get_dec_var_cust, read_penalty_coef, \
    import_value, export_data
from load import *
import csv
from datetime import datetime

total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
n_gen_csv = import_value("n_gen")
pop_size_csv = import_value("pop_size")
total_execution_csv = import_value("total_execution_time")

max_length = 0  # Initialize the maximum length of rows

for idx, (K_1, K_2, K_3, K_SC) in enumerate(coef_values, start=1):
    df = pd.read_csv(f"EXP_RES_X_{idx}.csv")
    df = df.iloc[:, 1:]
    results = []

    for index, row in df.iterrows():
        DEC_VARS = np.array([float(val) for val in row[:]]).reshape(13, T)
        RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
        ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
            POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, \
            IRG_DEF_SQTOT, PEN_SCALE = RESULT
        # Append the results after padding with np.nan
        padded_result = [PEN_MIN_SQTOT]
        max_length = max(max_length, len(padded_result))
        results.append(padded_result)

    # Pad the arrays in each row to make them consistent
    for i, row in enumerate(results):
        results[i] = row + [np.nan] * (max_length - len(row))

    # Export results to CSV
    export_data(np.array(results), f"PEN_MIN_SQTOT{idx}")

    print(f"Progress: {idx}/{total_iter}")

print("Progress completed.")
