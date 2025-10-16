import csv
import os
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp

from pathlib2 import Path
from scipy.ndimage import gaussian_filter1d

from load import *
from scipy.interpolate import interp1d

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Palatino Linotype"
sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Palatino Linotype', 'font.size': '10'})


def simulation_model(DEC_VARS_0, K_1, K_2, K_3, K_SC):
    STORAGES, RES_ELEV, RES_EVA, OUTFLOW, GEN_WATER, IRG_WATER, STORAGES_X, SPILLWAY = \
        calc_water_balance(DEC_VARS_0)

    POWER_MW, GEN_EN_GWH, POW_MW_AVE, SPILLWAY, GEN_WATER = \
        calc_energy_gen(GEN_WATER, EN_OPER_CAP, CONVERT, TUNNEL_DIA, TUNNEL_L, RES_COUNT, T,
                        RES_ELEV, OBS_EFF, TAILWAT_LEV, LOCAL_HEADLOSS_DEF, SPILLWAY, INST_CAP)

    PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, PEN_SCALE = \
        calc_penalty(S_MIN_ALL, S_MAX_ALL, S_MIN_SERIES, S_MAX_SERIES, S_0, STORAGES, STORAGES_X, K_1, K_2, K_3, K_SC, S_WEI)

    ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, IRG_DEF_SQTOT = \
        calc_objectives(OUTFLOW, OBS_NAT_FLOW, GEN_EN_GWH, IRG_WATER, PEN_ALL_TOTAL, PEN_MIN_SQTOT, PEN_MAX_SQTOT,
                        PEN_END_SQTOT, STO_DIF, PEN_SCALE)

    return ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
        POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE


def simulation_model_inflow_scenario(DEC_VARS_0, K_1, K_2, K_3, K_SC, inflow_scenario):
    NAT_INFLOW = get_data(inflow_scenario)

    STORAGES, RES_ELEV, RES_EVA, OUTFLOW, GEN_WATER, IRG_WATER, STORAGES_X, SPILLWAY = \
        calc_water_balance_scenario(DEC_VARS_0, NAT_INFLOW)

    POWER_MW, GEN_EN_GWH, POW_MW_AVE, SPILLWAY, GEN_WATER = \
        calc_energy_gen(GEN_WATER, EN_OPER_CAP, CONVERT, TUNNEL_DIA, TUNNEL_L, RES_COUNT, T,
                        RES_ELEV, OBS_EFF, TAILWAT_LEV, LOCAL_HEADLOSS_DEF, SPILLWAY, INST_CAP)

    PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, PEN_SCALE = \
        calc_penalty(S_MIN_ALL, S_MAX_ALL, S_MIN_SERIES, S_MAX_SERIES, S_0, STORAGES, STORAGES_X, K_1, K_2, K_3, K_SC, S_WEI)

    ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, IRG_DEF_SQTOT = \
        calc_objectives(OUTFLOW, OBS_NAT_FLOW, GEN_EN_GWH, IRG_WATER, PEN_ALL_TOTAL, PEN_MIN_SQTOT, PEN_MAX_SQTOT,
                        PEN_END_SQTOT, STO_DIF, PEN_SCALE)

    return ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
        POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE


def calc_water_balance_scenario(DEC_VARS_0, NAT_INFLOW):
    INFLOW = NAT_INFLOW - (CONS_OTHER + CONS_HIDR)
    ENG_NODES = [4, 5, 7, 8, 9]
    IRG_NODES = [1, 2, 3, 6, 10, 11, 12, 13]
    ENG_WATER = np.zeros((RES_COUNT, T))
    IRG_WATER = np.zeros((RES_COUNT, T))

    for i in [node - 1 for node in ENG_NODES]:
        ENG_WATER[i, :] = DEC_VARS_0[i, :]
    for i in [node - 1 for node in IRG_NODES]:
        IRG_WATER[i, :] = DEC_VARS_0[i, :]

    S_MAX_X = np.tile(S_MAX, (1, T))
    S_MAX_LEV_X = np.tile(S_MAX_LEVEL, (1, T))
    S_MIN_LEV_X = np.tile(S_MIN_LEVEL, (1, T))

    S_MIN_X = np.tile(S_MIN, (1, T))
    SPILL = np.zeros((RES_COUNT, T))
    INF = MX_CON_IRG @ (INFLOW - IRG_WATER)
    GEN_WATER = np.copy(ENG_WATER)
    REG_NODES = [5, 9]
    STORAGES = np.zeros((RES_COUNT, T))
    STORAGES = np.concatenate([S_0[:], STORAGES], 1)
    STORAGES_X = np.zeros((RES_COUNT, T + 1))
    SPILLAGE_CAP = get_data("SPILLAGE_CAP")

    ELV = np.zeros((RES_COUNT, T))
    RES_EVA = np.zeros((RES_COUNT, T))

    for t in range(T):
        for node in ENG_NODES:
            i = node - 1

            ELV[i, t] = vol_to_elev_idx((STORAGES[i, t] + INF[i, t]), VOL_ELV_COEF, i)
            RES_EVA[i, t] = elev_to_area_idx(ELV[i, t], ELV_ARE_COEF, EVA_MM, i, t)

            STORAGES[i, t + 1] = STORAGES[i, t] + INF[i, t] - RES_EVA[i, t] - (MX_ENR @ GEN_WATER)[i, t] - (MX_ENR @ SPILL)[i, t]
            STORAGES_X[i, t + 1] = STORAGES[i, t + 1]

            if STORAGES[i, t + 1] > S_MAX_X[i, t]:
                SPILL[i, t] += STORAGES[i, t + 1] - S_MAX_X[i, t]
                STORAGES[i, t + 1] = S_MAX_X[i, t]

            if STORAGES[i, t + 1] < S_MIN_X[i, t]:
                GEN_WATER[i, t] -= S_MIN_X[i, t] - STORAGES[i, t + 1]
                STORAGES[i, t + 1] = S_MIN_X[i, t]
                GEN_WATER[i, t] = np.maximum(GEN_WATER[i, t], 0)

            if node in REG_NODES:
                GEN_WATER[i, t] -= S_MAX_X[i, t] - STORAGES[i, t + 1]
                GEN_WATER[i, t] = np.maximum(GEN_WATER[i, t], 0)
                STORAGES[i, t + 1] = S_MAX_X[i, t]
            # else:
            #     if STORAGES[i, t + 1] < S_MIN_X[i, t] + OBS_EN_FIRM[i, t]*0.5:
            #         GEN_WATER[i, t] = 0
            #     else:
            #         GEN_WATER[i, t] = EN_FIRM[i, t]

            if GEN_WATER[i, t] > EN_OPER_CAP[i, t]:
                SPILL[i, t] += GEN_WATER[i, t] - EN_OPER_CAP[i, t]
                GEN_WATER[i, t] = EN_OPER_CAP[i, t]

    STORAGES = np.delete(STORAGES, 0, axis=1)
    STORAGES_X = np.delete(STORAGES_X, 0, axis=1)
    SPILLWAY = MX_ENR @ SPILL

    SPILLAGE_CAP = np.tile(SPILLAGE_CAP, (1, T))
    SPILLWAY = np.minimum(SPILLWAY, SPILLAGE_CAP)
    # Calculate reservoir elevations
    RES_ELEV = vol_to_elev((STORAGES), VOL_ELV_COEF)
    RES_ELEV = np.where(RES_ELEV < S_MIN_LEV_X, S_MIN_LEV_X, RES_ELEV)
    RES_ELEV = np.where(RES_ELEV > S_MAX_LEV_X, S_MAX_LEV_X, RES_ELEV)

    DELTA_S = MX_ENR @ (MX_CON @ (INFLOW - IRG_WATER) - (GEN_WATER + SPILLWAY) - (MX_CON @ RES_EVA))
    OUTFLOW = MX_CON @ (INFLOW - IRG_WATER - DELTA_S - RES_EVA)

    return STORAGES, RES_ELEV, RES_EVA, OUTFLOW, GEN_WATER, IRG_WATER, STORAGES_X, SPILLWAY


def calc_water_balance(DEC_VARS_0):
    ENG_NODES = [4, 5, 7, 8, 9]
    IRG_NODES = [1, 2, 3, 6, 10, 11, 12, 13]
    ENG_WATER = np.zeros((RES_COUNT, T))
    IRG_WATER = np.zeros((RES_COUNT, T))

    for i in [node - 1 for node in ENG_NODES]:
        ENG_WATER[i, :] = DEC_VARS_0[i, :]
    for i in [node - 1 for node in IRG_NODES]:
        IRG_WATER[i, :] = DEC_VARS_0[i, :]

    S_MAX_X = np.tile(S_MAX, (1, T))
    S_MAX_LEV_X = np.tile(S_MAX_LEVEL, (1, T))
    S_MIN_LEV_X = np.tile(S_MIN_LEVEL, (1, T))

    S_MIN_X = np.tile(S_MIN, (1, T))
    SPILL = np.zeros((RES_COUNT, T))
    INF = MX_CON_IRG @ (INFLOW - IRG_WATER)
    GEN_WATER = np.copy(ENG_WATER)
    REG_NODES = [5, 9]
    STORAGES = np.zeros((RES_COUNT, T))
    STORAGES = np.concatenate([S_0[:], STORAGES], 1)
    STORAGES_X = np.zeros((RES_COUNT, T + 1))
    SPILLAGE_CAP = get_data("SPILLAGE_CAP")

    ELV = np.zeros((RES_COUNT, T))
    RES_EVA = np.zeros((RES_COUNT, T))

    for t in range(T):
        for node in ENG_NODES:
            i = node - 1

            ELV[i, t] = vol_to_elev_idx((STORAGES[i, t] + INF[i, t]), VOL_ELV_COEF, i)
            RES_EVA[i, t] = elev_to_area_idx(ELV[i, t], ELV_ARE_COEF, EVA_MM, i, t)

            STORAGES[i, t + 1] = STORAGES[i, t] + INF[i, t] - RES_EVA[i, t] - (MX_ENR @ GEN_WATER)[i, t] - (MX_ENR @ SPILL)[i, t]
            STORAGES_X[i, t + 1] = STORAGES[i, t + 1]

            if STORAGES[i, t + 1] > S_MAX_X[i, t]:
                SPILL[i, t] += STORAGES[i, t + 1] - S_MAX_X[i, t]
                STORAGES[i, t + 1] = S_MAX_X[i, t]

            if STORAGES[i, t + 1] < S_MIN_X[i, t]:
                GEN_WATER[i, t] -= S_MIN_X[i, t] - STORAGES[i, t + 1]
                STORAGES[i, t + 1] = S_MIN_X[i, t]
                GEN_WATER[i, t] = np.maximum(GEN_WATER[i, t], 0)

            if node in REG_NODES:
                GEN_WATER[i, t] -= S_MAX_X[i, t] - STORAGES[i, t + 1]
                GEN_WATER[i, t] = np.maximum(GEN_WATER[i, t], 0)
                STORAGES[i, t + 1] = S_MAX_X[i, t]
            # else:
            #     if STORAGES[i, t + 1] < S_MIN_X[i, t] + OBS_EN_FIRM[i, t]*0.5:
            #         GEN_WATER[i, t] = 0
            #     else:
            #         GEN_WATER[i, t] = EN_FIRM[i, t]

            if GEN_WATER[i, t] > EN_OPER_CAP[i, t]:
                SPILL[i, t] += GEN_WATER[i, t] - EN_OPER_CAP[i, t]
                GEN_WATER[i, t] = EN_OPER_CAP[i, t]

    STORAGES = np.delete(STORAGES, 0, axis=1)
    STORAGES_X = np.delete(STORAGES_X, 0, axis=1)
    SPILLWAY = MX_ENR @ SPILL

    SPILLAGE_CAP = np.tile(SPILLAGE_CAP, (1, T))
    SPILLWAY = np.minimum(SPILLWAY, SPILLAGE_CAP)
    # Calculate reservoir elevations
    RES_ELEV = vol_to_elev((STORAGES), VOL_ELV_COEF)
    RES_ELEV = np.where(RES_ELEV < S_MIN_LEV_X, S_MIN_LEV_X, RES_ELEV)
    RES_ELEV = np.where(RES_ELEV > S_MAX_LEV_X, S_MAX_LEV_X, RES_ELEV)

    DELTA_S = MX_ENR @ (MX_CON @ (INFLOW - IRG_WATER) - (GEN_WATER + SPILLWAY) - (MX_CON @ RES_EVA))
    OUTFLOW = MX_CON @ (INFLOW - IRG_WATER - DELTA_S - RES_EVA)

    return STORAGES, RES_ELEV, RES_EVA, OUTFLOW, GEN_WATER, IRG_WATER, STORAGES_X, SPILLWAY


def calc_energy_gen(GEN_WATER, EN_OPER_CAP, CONVERT, TUNNEL_DIA, TUNNEL_L, RES_COUNT, T, RES_ELEV, OBS_EFF, TAILWAT_LEV, LOCAL_HEADLOSS_DEF,
                    SPILLWAY, INST_CAP):
    # EN_SECN = np.maximum(GEN_WATER - OBS_EN_FIRM, 0)
    # EN_FIRM = np.maximum(OBS_EN_FIRM, GEN_WATER)

    # Calculate tunnel discharge flow rate
    TUNNEL_FLOW_CMS = GEN_WATER * 1e6 / (30 * 24 * 60 * 60)  # np.array(DAYS_IN_MONTHS)

    # Calculate hydraulic gradient for tunnel discharge with condition to handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        HYDRAULIC_GRADIENT = np.where(TUNNEL_DIA != 0, (0.00202 * (TUNNEL_FLOW_CMS ** 2) / (TUNNEL_DIA ** (16 / 3))), 0)

    # Calculate head loss due to tunnel discharge
    TUNNEL_HEADLOSS = HYDRAULIC_GRADIENT * TUNNEL_L

    # # Define rows to zero out in the identity matrix
    # rows_to_zero_out = [0, 1, 2, 5, 9, 10, 11, 12]
    # # Create an identity matrix and zero out specified rows
    # # Calculate local head loss matrix
    # identity_matrix = np.eye(RES_COUNT)
    # identity_matrix[rows_to_zero_out] = 0
    # LOCAL_HEADLOSS = np.dot(identity_matrix, np.ones((RES_COUNT, T)) * LOCAL_HEADLOSS_DEF)
    LOCAL_HEADLOSS = 0

    # Repeat energy production constraint for each time period
    TAILWAT_LEV = np.tile(TAILWAT_LEV, T)

    # Calculate net head at the turbine
    NET_HEAD = (RES_ELEV - TAILWAT_LEV - TUNNEL_HEADLOSS - LOCAL_HEADLOSS)

    # Calculate power output in MW
    POWER_MW = (1000 * C_GRAVITY * OBS_EFF * NET_HEAD * TUNNEL_FLOW_CMS) / 1e6
    INST_CAP = np.tile(INST_CAP, (1, T))
    POWER_MW = np.minimum(POWER_MW, INST_CAP)
    # EN_SECN_MW = np.maximum(POWER_MW - OBS_EN_FIRM_MW, 0)
    # EN_FIRM_MW = np.maximum(OBS_EN_FIRM_MW, POWER_MW)

    # Calculate energy production in GWh
    GEN_EN_GWH = POWER_MW * 24 * 30 / 1000  # np.array(DAYS_IN_MONTHS)
    # EN_SECN_GWH = np.maximum(GEN_EN_GWH - OBS_EN_FIRM_GWH, 0)
    # EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, GEN_EN_GWH)

    # Calculate average power output
    POW_MW_AVE = np.average(POWER_MW)
    # POW_MW_FIRM_AVE = np.average(EN_FIRM_MW)
    # POW_MW_SECN_AVE = np.average(EN_SECN_MW)

    return POWER_MW, GEN_EN_GWH, POW_MW_AVE, SPILLWAY, GEN_WATER


def calc_regulation_ratio(OUTFLOW, ENV_MNG_CLASS, OBS_NAT_FLOW):
    # Calculate deviation from the natural flow for each node and month
    ECO_DEV = np.sum(((OUTFLOW - ENV_MNG_CLASS) / np.max(ENV_MNG_CLASS, axis=1, keepdims=True)) ** 2, axis=1)

    # # Calculate regulation ratio for each node
    # REG_RATIO = np.sum(OUTFLOW, axis=1) / np.sum(OBS_NAT_FLOW, axis=1)
    # # Calculate flow alteration ratio for the basin output (last node)
    # last_node_outflow = OUTFLOW[-1, :]  # Outflow of the last node
    # last_node_nat_flow = OBS_NAT_FLOW[-1, :]  # Natural flow of the last node
    # REG_RATIO = np.sum(last_node_outflow) / np.sum(last_node_nat_flow)

    # Aggregate results over all nodes for the whole basin
    # AVE_REG_RATIO = np.mean(REG_RATIO)  # Changed from np.average to np.mean for clarity
    AVE_REG_RATIO = ECO_DEV[-1]  # Basin Outlet deviation for scenario analysis
    TOT_ECO_DEV = np.sum(ECO_DEV)  # Summing the deviations across all nodes

    return AVE_REG_RATIO, TOT_ECO_DEV


def calc_penalty(S_MIN_ALL, S_MAX_ALL, S_MIN_SERIES, S_MAX_SERIES, S_0, STORAGES, STORAGES_X, K_1, K_2, K_3, K_SC, S_WEI):
    ENG_NODES = [4, 7, 8]  # [4, 7, 8]
    E = [node - 1 for node in ENG_NODES]  # Adjust for 0-based indexing in numpy
    STORAGES_E = STORAGES_X[E]  # STORAGES_X: Storages with exceeding boundaries
    S_END = STORAGES_E[:, -1]
    S_INI = S_0[E].flatten()  # S_INIT AVERAGE
    S_WEI_E = 1  # S_WEI[E]  # S_WEI[E] PROP. WITH ACTIVE VOLUME
    S_MIN_SERIES_E = np.tile(S_MIN, (1, 12))[E]  # S_MIN_ALL[E]  # S_MIN_ALL S_MIN_SERIES
    S_MAX_SERIES_E = np.tile(S_MAX, (1, 12))[E]  # S_MAX_ALL[E]  # S_MAX_ALL S_MAX_SERIES
    S_INI_E = np.tile(S_INI, (1, 12))
    OBS_STO_E = OBS_STO[E]
    STO_DIF = np.sum(np.square((STORAGES_E - OBS_STO_E)))

    PEN_MIN = np.maximum(0, ((S_MIN_SERIES_E - STORAGES_E) * S_WEI_E))
    PEN_MAX = np.maximum(0, ((STORAGES_E - S_MAX_SERIES_E) * S_WEI_E))
    PEN_END = np.maximum(0, (S_INI - S_END)) * S_WEI_E

    PEN_MIN_SQTOT = K_1 * np.sum(np.square(PEN_MIN))
    PEN_MAX_SQTOT = K_2 * np.sum(np.square(PEN_MAX))
    PEN_END_SQTOT = K_3 * np.sum(np.square(PEN_END))

    PEN_ALL_TOTAL = PEN_MIN_SQTOT + PEN_MAX_SQTOT + PEN_END_SQTOT
    PEN_SCALE = K_SC * PEN_ALL_TOTAL
    # X_S_END = OBS_STO_E[:, -1]
    # X_PEN_MIN = np.minimum(0, ((S_MIN_SERIES_E - OBS_STO_E) * S_WEI_E))  # S_MIN_ALL S_MIN_SERIES
    # X_PEN_MAX = np.minimum(0, ((OBS_STO_E - S_MAX_SERIES_E) * S_WEI_E))  # S_MAX_ALL S_MAX_SERIES
    # X_PEN_END = ((S_INI - X_S_END) * S_WEI_E)
    # X_PEN_MIN_SQTOT = np.sum(np.square(X_PEN_MIN))
    # X_PEN_MAX_SQTOT = np.sum(np.square(X_PEN_MAX))
    # X_PEN_END_SQTOT = np.sum(np.square(X_PEN_END))
    #
    # C_1 =  1/(X_PEN_MIN_SQTOT / PEN_MIN_SQTOT)
    # C_2 =  1/(X_PEN_MAX_SQTOT / PEN_MAX_SQTOT)
    # C_3 =  1/(X_PEN_END_SQTOT / PEN_END_SQTOT)
    # sorti = np.average((S_MAX_SERIES_E - S_MIN_SERIES_E) / 2, axis=1)

    return PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, PEN_SCALE


def calc_objectives(OUTFLOW, OBS_NAT_FLOW, GEN_EN_GWH, IRG_WATER, PEN_ALL_TOTAL, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, STO_DIF, PEN_SCALE):
    # (1) Energy objective: maximize power generation (f_PGEN).
    ENG_TOT = np.sum(GEN_EN_GWH)
    GWH_CAP_TOT = np.sum(GWH_CAP)
    ENG_DEF_SQTOT = ((GWH_CAP_TOT - ENG_TOT) / GWH_CAP_TOT) ** 2

    # (2) Food objective: minimize the irrigation requirements (f_IRDEF)
    IRG_NODES = [1, 2, 3, 6, 10, 11, 12, 13]
    I = [node - 1 for node in IRG_NODES]  # Adjust for 0-based indexing in numpy
    IRG_DEF_SQTOT = np.sum(((IR_DEMAND[I] - IRG_WATER[I]) / np.max(IR_DEMAND[I], axis=0)) ** 2)
    IRG_TOT = np.sum(IRG_WATER[I])

    # (3) Ecosystem objective: minimizing the deviation from the natural flow (f_ECODEV)
    AVE_REG_RATIO, TOT_ECO_DEV = calc_regulation_ratio(OUTFLOW, NAT_EMC, OBS_NAT_FLOW)
    # MENO = np.mean(NAT_EMC_A, axis=1, keepdims=True)

    # Proportions Coefficients to normalize and scale the impact of different objectives on the final objective functions.
    P_1 = 1  # ENG_DEF_SQTOT/ENG_DEF_SQTOT
    P_2 = IRG_DEF_SQTOT / ENG_DEF_SQTOT
    P_3 = TOT_ECO_DEV / ENG_DEF_SQTOT

    # Objective Functions combining the original measures with penalty terms scaled by PEN_SCALE.
    ENR_OBJ = ENG_DEF_SQTOT + PEN_SCALE * P_1
    IRG_OBJ = IRG_DEF_SQTOT + PEN_SCALE * P_2
    ECO_OBJ = TOT_ECO_DEV + PEN_SCALE * P_3

    return ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, IRG_DEF_SQTOT


def vol_to_elev_idx(volume, coefficients, index):
    a = coefficients[index, 0]
    b = coefficients[index, 1]
    c = coefficients[index, 2]

    # Replace negative and zero volume values with a small positive value
    volume = np.where(volume <= 0, 1e-10, volume)

    # Calculate elevation (m) using logarithms to avoid invalid power operation
    y = a * np.exp(b * np.log(volume)) + c

    # Replace NaN values with 0
    elevation = np.nan_to_num(y, nan=0)

    return elevation


def vol_to_elev(volume, coefficients):
    a = coefficients[:, 0]
    b = coefficients[:, 1]
    c = coefficients[:, 2]

    # Replace negative and zero volume values with a small positive value
    volume = np.where(volume <= 0, 1e-10, volume)

    # Calculate elevation (m) using logarithms to avoid invalid power operation
    y = a[:, np.newaxis] * np.exp(b[:, np.newaxis] * np.log(volume)) + c[:, np.newaxis]

    # Replace NaN values with 0
    elevation = np.nan_to_num(y, nan=0)

    return elevation

def elev_to_area_idx(elevation, coefficients, eva_mm, index, month):
    a = coefficients[index, 0]
    b = coefficients[index, 1]

    # Replace negative and zero elevation values (m) with a small positive value
    elevation = np.where(elevation <= 0, 1e-10, elevation)

    # Calculate area (sq km)
    y = a * (elevation ** b)
    area = np.nan_to_num(y, nan=0)

    # Calculate evaporation (hm3) based on the calculated area and eva_mm
    evaporation = area * eva_mm[index, month] * 10e-4

    # Ensure evaporation is non-negative
    evaporation = np.maximum(evaporation, 0)

    return evaporation


def elev_to_area(elevation, coefficients, eva_mm):
    a = coefficients[:, 0]
    b = coefficients[:, 1]

    # Replace negative and zero elevation values (m) with a small positive value
    elevation = np.where(elevation <= 0, 1e-10, elevation)

    # Calculate area (sq km)
    y = a[:, np.newaxis] * (elevation ** b[:, np.newaxis])
    area = np.nan_to_num(y, nan=0)

    # Calculate evaporation (hm3) based on the calculated area and eva_mm
    evaporation = area * eva_mm * 10e-4
    # Ensure evaporation is non-negative
    evaporation = np.maximum(evaporation, 0)

    return evaporation


def calc_performance_indicators(hist_F, n_pop, F, total_execution_time, n_evals):
    """
    Calculates NDSGR, UPM, SPM, and MCT performance indicators.

    Args:
        hist_F (list): A list of Pareto fronts at each generation.
        n_pop (int): The total population size.
        F (np.array): The final Pareto front.
        total_execution_time (str): The total execution time in the format "MM:SS".
        n_evals (np.array): The number of function evaluations at each generation.

    Returns:
        tuple: A tuple containing NDSGR, UPM, SPM, and MCT values.
    """

    n_gen = len(hist_F)
    non_dominated_sizes = [len(F) for F in hist_F]

    # Calculate NDSGR
    ndsg_rates = [
        (non_dominated_sizes[i]) / n_pop
        for i in range(n_gen - 1)
    ]
    ndsgr = np.mean(ndsg_rates)

    # Calculate UPM (using crowding distance as a uniformity measure)
    n_points, n_obj = F.shape

    di_list = []

    for i in range(n_points):
        distances = np.linalg.norm(F[i, :] - F, axis=1)  # Calculate distances to all other points

        di = np.min(distances[np.arange(n_points) != i])  # Minimum distance to other points

        di_list.append(di)

    di_mean = np.mean(di_list)  # Mean of minimum distances

    upm = np.sqrt(np.sum((di_list - di_mean) ** 2) / (n_points - 1))

    # Calculate SPM (example using the range of objective values)
    ranges = [np.ptp(F, axis=0) for F in hist_F]  # ptp calculates range (max - min)
    spm = np.mean([np.mean(r) for r in ranges])

    # Calculate MCT
    hours, minutes, seconds = map(int, total_execution_time.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds

    # Calculate MCT
    # mct = total_seconds / n_evals[-1]  # Divide by the total number of evaluations
    mct = total_seconds / n_gen  # Divide by tot gen
    return ndsgr, upm, spm, mct


def x_normalize_data(value, indicator, indicator_type):
    """Normalize the data based on a predefined max value."""

    indicator_max_values = {
        'HV': 3.85,
        'UPM': 0.01265,
        'SPM': 1.2529,
        'NDSGR': 1,
        'MCT': 6.6
    }

    max_val = indicator_max_values[indicator]

    # Handle the case where all values are the same (no variation)
    if indicator_type == 'maximize':
        # If we want to maximize, the higher the value, the closer to 1
        return 1 - (value / max_val)
    elif indicator_type == 'minimize':
        # If we want to minimize, the lower the value, the closer to 0
        return value / max_val


def x_compare_models_polar(data, indicators, indicator_types):
    """Compare two models on a polar chart with normalization."""
    num_vars = len(indicators)

    # Split the data for both models (they are dictionaries)
    nsga2_data = data['NSGA-II']
    nsga3_data = data['NSGA-III']

    # Normalize the data for both models, separately per indicator
    nsga2_data_normalized = [
        normalize_data(nsga2_data[indicator], indicator, indicator_types[i])
        for i, indicator in enumerate(indicators)
    ]
    nsga3_data_normalized = [
        normalize_data(nsga3_data[indicator], indicator, indicator_types[i])
        for i, indicator in enumerate(indicators)
    ]

    # Compute angle for each indicator
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the loop of the radar chart (the first value is repeated at the end)
    nsga2_data_normalized.append(nsga2_data_normalized[0])  # Closing the loop for NSGA-II
    nsga3_data_normalized.append(nsga3_data_normalized[0])  # Closing the loop for NSGA-III
    angles.append(angles[0])  # Closing the loop for angles

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot both models
    ax.plot(angles, nsga2_data_normalized, linewidth=2, linestyle='solid', label='NSGA-II', color='b')
    ax.plot(angles, nsga3_data_normalized, linewidth=2, linestyle='solid', label='NSGA-III', color='r')

    # Fill the areas for better comparison (transparency to show overlapping regions)
    ax.fill(angles, nsga2_data_normalized, color='b', alpha=0.25)
    ax.fill(angles, nsga3_data_normalized, color='r', alpha=0.25)

    # Set the labels for each indicator
    ax.set_yticklabels([])  # Hide the radial labels (values)
    ax.set_xticks(angles[:-1])  # Set angle ticks (indicator names)
    ax.set_xticklabels(indicators, fontsize=12)  # Set indicator names

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Show the plot with a title
    plt.title('Normalized Polar Chart Comparison: NSGA-II vs NSGA-III', size=14)
    plt.show()


def plot_hypervolume(Hypervolume, n_evals, F, hist_F):
    """
    Calculates and plots the hypervolume convergence.

    Args:
        n_evals (np.array): The number of function evaluations at each generation.
        F (np.array): The final Pareto front.
        hist_F (list): A list of Pareto fronts at each generation.
    """

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    # Adjust the reference point calculation
    # ref_point = approx_nadir * 1.1  # Increase nadir by 10%
    worst_point = F.max(axis=0)
    ref_point = worst_point + (worst_point - approx_ideal) * 0.1
    # ref_point = [0.8, 1.0, 15.0]

    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='crimson', lw=0.8)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='crimson', marker="o", s=5, alpha=0.7)
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume (HV)")
    plt.show()

def xcorner_evaluation(F, worst_point):
    """
    Performs corner evaluation to update the nadir point.

    Args:
        F (np.ndarray): The Pareto front approximation (N × M).
        worst_point (np.ndarray): The current worst estimated nadir point (M,).

    Returns:
        np.ndarray: The updated nadir point.
    """

    num_objectives = F.shape[1]  # M = number of objectives (3 in your case)
    nadir_point = worst_point.copy()

    # Compute minimum of each objective (across all solutions in F)
    min_vals = np.min(F, axis=0)
    print(f"Minimum values across objectives in F: {min_vals}")

    for i in range(num_objectives):
        # Create a corner point with min values and adjust the i-th objective
        corner_point = min_vals.copy()
        corner_point[i] = np.max(F[:, i])  # Adjust the i-th objective to the maximum in F

        print(f"Corner point for objective {i}: {corner_point}")
        print(f"Current Nadir Point: {nadir_point}")

        # Update nadir point if the corner point has worse values
        nadir_point = np.maximum(nadir_point, corner_point)
        print(f"Updated Nadir Point: {nadir_point}")

    return nadir_point




def plot_hypervolume_compare(Hypervolume, n_evals, F1_name, F1, hist_F1, F2_name, F2, hist_F2):
    approx_ideal = np.min([F1.min(axis=0), F2.min(axis=0)], axis=0)
    approx_nadir = np.max([F1.max(axis=0), F2.max(axis=0)], axis=0)
    worst_point = approx_nadir
    ref_point = worst_point + (worst_point - approx_ideal) * 0.1

    color_1 = "#0D95D0"  # Teal
    color_2 = "#E72F52"  # Red

    hv1 = [
        Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            ideal=F1.min(axis=0),
            nadir=F1.max(axis=0),
        ).do(_F)
        for _F in hist_F1
    ]

    hv2 = [
        Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            ideal=F2.min(axis=0),
            nadir=F2.max(axis=0),
        ).do(_F)
        for _F in hist_F2
    ]

    export_data(hv1, "hv1")
    export_data(hv2, "hv2")

    # hv1 = get_data("EXP_hv1")
    # hv2 = get_data("EXP_hv2")

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv1, color=color_1, lw=1, alpha=0.5, label=F1_name)
    plt.plot(n_evals, hv2, color=color_2, lw=1, alpha=0.5, label=F2_name)
    plt.scatter(n_evals, hv1, edgecolor="none", facecolor=color_1, marker="o", s=3, alpha=0.8)
    plt.scatter(n_evals, hv2, edgecolor="none", facecolor=color_2, marker="o", s=3, alpha=0.8)

    plt.title("Hypervolume Convergence")
    plt.xlabel("Function Evaluations (K)")
    plt.xticks(np.arange(200000, n_evals[-1] + 1, 200000),
               [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)])
    plt.ylabel("Hypervolume (HV)")
    plt.legend()
    plt.show()


def BCCplot_hypervolume_compare_inputhv(n_evals, F1_name, F2_name):
    color_1 = "#008080"  # Teal
    color_2 = "#f44336"  # Red

    hv1 = get_data("EXP_hv1")
    hv2 = get_data("EXP_hv2")

    plt.figure(figsize=(7, 5))

    # Plot actual lines with alpha transparency
    plt.plot(n_evals, hv1, color=color_1, lw=2, alpha=0.3)
    plt.plot(n_evals, hv2, color=color_2, lw=2, alpha=0.3)

    # Scatter points
    plt.scatter(n_evals, hv1, facecolor=color_1, edgecolor="none", marker="o", s=4, alpha=0.9)
    plt.scatter(n_evals, hv2, facecolor=color_2, edgecolor="none", marker="o", s=4, alpha=0.9)

    # Create invisible lines for a separate legend (full opacity)
    legend_line1, = plt.plot([], [], color=color_1, lw=2, label="NSGA-II")
    legend_line2, = plt.plot([], [], color=color_2, lw=2, label="NSGA-III")

    # Title and labels
    # plt.title("Hypervolume Convergence", fontsize=14, fontweight="bold")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume (HV)")

    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    # Grid settings
    plt.grid(True, linestyle="dotted", alpha=0.5)

    # Add separate legend (with opaque lines)
    plt.legend(handles=[legend_line1, legend_line2], fontsize=11, frameon=True, loc="best")

    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_hypervolume_compare_inputhv(n_evals, F1_name, F2_name, normalize=False):
    color_1 = "#0D95D0"  # Blue for NSGA-II
    color_2 = "#E72F52"  # Red for NSGA-III

    faded_color_1 = "#AACBE9"
    faded_color_2 = "#F4ADA8"

    hv1 = get_data("EXP_hv1")  # NSGA-II
    hv2 = get_data("EXP_hv2")  # NSGA-III

    hv1 = np.ravel(hv1)  # Flatten (1000,1) → (1000,)
    hv2 = np.ravel(hv2)
    n_evals = np.ravel(n_evals)

    # Normalize the hypervolume series if specified
    if normalize:
        hv1 = (hv1 - hv1.min()) / (hv1.max() - hv1.min())
        hv2 = (hv2 - hv2.min()) / (hv2.max() - hv2.min())

    plt.figure(figsize=(7, 5))


    # Plot actual lines with alpha transparency
    plt.plot(n_evals, hv1, color=color_1, lw=1.2, alpha=0.9)
    plt.plot(n_evals, hv2, color=color_2, lw=1.2, alpha=0.9)

    # Fill area between the curves and x-axis
    plt.fill_between(n_evals, hv1, hv2, color=faded_color_1, alpha=0.6, zorder=1)  # Fill NSGA-II to x-axis
    plt.fill_between(n_evals, hv2, 0, color=faded_color_2,   alpha=0.6, zorder=1)  # Fill NSGA-III to x-axis

    # Calculate the area under each curve
    area_hv1 = np.trapz(hv1, n_evals)
    area_hv2 = np.trapz(hv2, n_evals)

    print(f"Area under NSGA-II curve: {area_hv1:.2f}")
    print(f"Area under NSGA-III curve: {area_hv2:.2f}")

    # Create invisible lines for a separate legend (full opacity)
    legend_line1, = plt.plot([], [], color=color_1, lw=2, label="NSGA-II")
    legend_line2, = plt.plot([], [], color=color_2, lw=2, label="NSGA-III")

    # Title and labels
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume (HV)")

    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    # Grid settings
    plt.grid(True, linestyle="dotted", alpha=0.5)

    # Add separate legend (with opaque lines)
    plt.legend(handles=[legend_line1, legend_line2], fontsize=11, frameon=True, loc="best")

    # Show the area difference annotation on the plot
    plt.text(n_evals[len(n_evals) // 2], max(hv1 + hv2) * 0.9,
             f"Δ Area: {abs(area_hv1 - area_hv2):.2f}",
             fontsize=12, color="black", ha="center")

    plt.show()

def plot_hypervolume_compare_inputhv_range(n_evals, F1_name, F2_name, normalize=False):
    color_1 = "#0D95D0"  # Blue for NSGA-II
    color_2 = "#E72F52"  # Red for NSGA-III

    faded_color_1 = "#AACBE9"
    faded_color_2 = "#F4ADA8"

    # Assume get_data returns a list of 5 runs, each a (1000, 1) array
    hv1_runs = [get_data_metrichist(f"EXP_hv_nsgs2_seed{i+1}") for i in range(5)] # NSGA-II
    hv2_runs = [get_data_metrichist(f"EXP_hv_nsgs3_seed{i+1}") for i in range(5)] # NSGA-III

    hv1_runs_flat = [np.ravel(run) for run in hv1_runs]
    hv2_runs_flat = [np.ravel(run) for run in hv2_runs]

    n_evals = np.ravel(n_evals)

    # Normalize each run individually if specified
    if normalize:
        for i in range(5):
            hv1_runs_flat[i] = (hv1_runs_flat[i] - hv1_runs_flat[i].min()) / (hv1_runs_flat[i].max() - hv1_runs_flat[i].min())
            hv2_runs_flat[i] = (hv2_runs_flat[i] - hv2_runs_flat[i].min()) / (hv2_runs_flat[i].max() - hv2_runs_flat[i].min())

    # Calculate min, max, and mean for each evaluation point across runs
    hv1_min = np.min(np.array(hv1_runs_flat), axis=0)
    hv1_max = np.max(np.array(hv1_runs_flat), axis=0)
    hv1_mean = np.mean(np.array(hv1_runs_flat), axis=0)

    hv2_min = np.min(np.array(hv2_runs_flat), axis=0)
    hv2_max = np.max(np.array(hv2_runs_flat), axis=0)
    hv2_mean = np.mean(np.array(hv2_runs_flat), axis=0)

    plt.figure(figsize=(7, 5),  dpi=200)

    # Plot mean lines
    plt.plot(n_evals, hv1_mean, color=color_1, lw=1.2, alpha=0.9, label="NSGA-II")
    plt.plot(n_evals, hv2_mean, color=color_2, lw=1.2, alpha=0.9, label="NSGA-III")

    # # Use the first series as the main line.
    # hv1_series1 = hv1_runs_flat[0]
    # hv2_series1 = hv2_runs_flat[0]
    # plt.plot(n_evals, hv1_series1, color="orange", lw=1.2, alpha=0.9, label="NSGA-II (Mean)")
    # plt.plot(n_evals, hv2_series1, color="purple", lw=1.2, alpha=0.9, label="NSGA-III (Mean)")

    # Fill area between min and max
    plt.fill_between(n_evals, hv1_min, hv1_max, color=faded_color_1, alpha=0.5)
    plt.fill_between(n_evals, hv2_min, hv2_max, color=faded_color_2, alpha=0.5)

    # # Calculate the area under the mean curves
    # area_hv1 = np.trapz(hv1_mean, n_evals)
    # area_hv2 = np.trapz(hv2_mean, n_evals)

    # print(f"Area under NSGA-II (Mean) curve: {area_hv1:.2f}")
    # print(f"Area under NSGA-III (Mean) curve: {area_hv2:.2f}")

    # Title and labels
    plt.xlabel("Function Evaluations")
    plt.ylabel("HV - Hypervolume")

    # Y-axis formatting to 0.0
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    plt.gca().set_xlim(-10, n_evals[-1] * 1.01)
    plt.gca().set_ylim(-0.1, max(hv1_max.max(), hv2_max.max()) * 1.1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    # Grid settings
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add legend
    plt.legend(fontsize=11, frameon=True, loc="lower right")

    # Show the area difference annotation on the plot
    # plt.text(n_evals[len(n_evals) // 2], max(hv1_max.max(), hv2_max.max()) * 0.9,
             # f"Δ Area: {abs(area_hv1 - area_hv2):.2f}",
             # fontsize=12, color="black", ha="center")

    plt.show()

import matplotlib.ticker as ticker
def plot_hypervolume_compare_inputhv_range_turkish(n_evals, F1_name, F2_name, normalize=False):
    """
    Plots the Hypervolume (HV) progression for NSGA-II and NSGA-III over
    Function Evaluations, showing the mean, min, and max range for multiple runs.

    Args:
        n_evals (numpy.ndarray): Array of function evaluation counts (x-axis data).
        F1_name (str): Placeholder for first algorithm name (NSGA-II).
        F2_name (str): Placeholder for second algorithm name (NSGA-III).
        normalize (bool): Whether to normalize the HV data for each run.
    """
    color_1 = "#0D95D0"  # Blue for NSGA-II
    color_2 = "#E72F52"  # Red for NSGA-III

    faded_color_1 = "#AACBE9"
    faded_color_2 = "#F4ADA8"

    # Load data for 5 runs for each algorithm
    hv1_runs = [get_data_metrichist(f"EXP_hv_nsgs2_seed{i + 1}") for i in range(5)]  # NSGA-II
    hv2_runs = [get_data_metrichist(f"EXP_hv_nsgs3_seed{i + 1}") for i in range(5)]  # NSGA-III

    hv1_runs_flat = [np.ravel(run) for run in hv1_runs]
    hv2_runs_flat = [np.ravel(run) for run in hv2_runs]

    n_evals = np.ravel(n_evals)

    # Normalize each run individually if specified
    if normalize:
        for i in range(5):
            hv1_runs_flat[i] = (hv1_runs_flat[i] - hv1_runs_flat[i].min()) / (hv1_runs_flat[i].max() - hv1_runs_flat[i].min())
            hv2_runs_flat[i] = (hv2_runs_flat[i] - hv2_runs_flat[i].min()) / (hv2_runs_flat[i].max() - hv2_runs_flat[i].min())

    # Calculate min, max, and mean for each evaluation point across runs
    hv1_min = np.min(np.array(hv1_runs_flat), axis=0)
    hv1_max = np.max(np.array(hv1_runs_flat), axis=0)
    hv1_mean = np.mean(np.array(hv1_runs_flat), axis=0)

    hv2_min = np.min(np.array(hv2_runs_flat), axis=0)
    hv2_max = np.max(np.array(hv2_runs_flat), axis=0)
    hv2_mean = np.mean(np.array(hv2_runs_flat), axis=0)

    plt.figure(figsize=(7, 5), dpi=200)

    # Plot mean lines
    plt.plot(n_evals, hv1_mean, color=color_1, lw=1.2, alpha=0.9, label="NSGA-II")
    plt.plot(n_evals, hv2_mean, color=color_2, lw=1.2, alpha=0.9, label="NSGA-III")

    # Fill area between min and max (Range)
    plt.fill_between(n_evals, hv1_min, hv1_max, color=faded_color_1, alpha=0.5)
    plt.fill_between(n_evals, hv2_min, hv2_max, color=faded_color_2, alpha=0.5)

    # Title and labels
    plt.xlabel("Jenerasyon")
    plt.ylabel("HV - Hiper Hacim")

    # --- FORMATTING ---

    # Y-axis formatting: Use COMMA (,) as decimal separator and 1 decimal place.
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:,.1f}'.format(y).replace('.', ',')))

    # X-axis formatting: Use POINT (.) as thousands separator and no decimals.
    # We use a trick: format with Python default ',', then replace ',' with '.'
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '{:,.0f}'.format(x).replace(',', '_TEMP_').replace('.', ',').replace('_TEMP_', '.')
    ))

    # Set plot limits
    plt.gca().set_xlim(-10, n_evals[-1] * 1.01)
    # plt.gca().set_ylim(-0.1, max(hv1_max.max(), hv2_max.max()) * 1.1)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Grid settings
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add legend
    plt.legend(fontsize=11, frameon=True, loc="lower right")

    plt.show()


def plot_igdplus_compare_inputigdplus_range(n_evals, F1_name, F2_name, normalize=False):
    color_1 = "#0D95D0"  # Teal for NSGA-II
    color_2 = "#E72F52"  # Red for NSGA-III

    faded_color_1 = "#AACBE9"
    faded_color_2 = "#F4ADA8"

    # Assume get_data returns a list of 5 runs, each a (1000, 1) array
    igd1_runs = [get_data_metrichist(f"EXP_igdplus_nsgs2_seed{i+1}") for i in range(5)] # NSGA-II
    igd2_runs = [get_data_metrichist(f"EXP_igdplus_nsgs3_seed{i+1}") for i in range(5)] # NSGA-III

    igd1_runs_flat = [np.ravel(run) for run in igd1_runs]
    igd2_runs_flat = [np.ravel(run) for run in igd2_runs]

    n_evals = np.ravel(n_evals)

    # Normalize each run individually if specified
    if normalize:
        for i in range(5):
            igd1_runs_flat[i] = (igd1_runs_flat[i] - igd1_runs_flat[i].min()) / (igd1_runs_flat[i].max() - igd1_runs_flat[i].min())
            igd2_runs_flat[i] = (igd2_runs_flat[i] - igd2_runs_flat[i].min()) / (igd2_runs_flat[i].max() - igd2_runs_flat[i].min())

    # Calculate min, max, and mean for each evaluation point across runs
    igd1_min = np.min(np.array(igd1_runs_flat), axis=0)
    igd1_max = np.max(np.array(igd1_runs_flat), axis=0)
    igd1_mean = np.mean(np.array(igd1_runs_flat), axis=0)

    igd2_min = np.min(np.array(igd2_runs_flat), axis=0)
    igd2_max = np.max(np.array(igd2_runs_flat), axis=0)
    igd2_mean = np.mean(np.array(igd2_runs_flat), axis=0)

    plt.figure(figsize=(7, 5),dpi=200)

    # Plot mean lines
    plt.plot(n_evals, igd1_mean, color=color_1, lw=1.2, alpha=0.9, label="NSGA-II")
    plt.plot(n_evals, igd2_mean, color=color_2, lw=1.2, alpha=0.9, label="NSGA-III")

    # Fill area between min and max
    plt.fill_between(n_evals, igd1_min, igd1_max, color=faded_color_1, alpha=0.5)
    plt.fill_between(n_evals, igd2_min, igd2_max, color=faded_color_2, alpha=0.5)

    # Title and labels
    plt.xlabel("Function Evaluations")
    plt.ylabel(r"IGD$\mathrm{}^{+}$ - Inverted Generational Distance Plus")

    # Y-axis formatting to 0.0
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    plt.yscale("log")
    # Grid settings
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add legend
    plt.legend(fontsize=11, frameon=True, loc="upper right")

    plt.show()

def get_data_hist(file_name):
    csv_name_path = PATHWAY /"METRIC_HIST" /(file_name + ".csv")
    data = np.genfromtxt(csv_name_path, delimiter=",", skip_header=1)
    # Check if the array has only one dimension
    if data.ndim == 1:
        # For one-dimensional array, remove the first element
        data = data[1:].reshape(1, -1)
    else:
        # For two-dimensional array, skip the first column
        data = data[:, 1:]

    return data


def plot_hypervolume_compare_BCK(Hypervolume, n_evals, F1_name, F1, hist_F1, F2_name, F2, hist_F2):
    """
    Calculates and plots the hypervolume convergence for two Pareto front sets.

    Args:
        Hypervolume (pymoo.indicators.hv.Hypervolume): The Hypervolume class.
        n_evals (np.array): The number of function evaluations at each generation.
        F1 (np.array): The final Pareto front for the first model.
        hist_F1 (list): A list of Pareto fronts at each generation for the first model.
        F2 (np.array): The final Pareto front for the second model.
        hist_F2 (list): A list of Pareto fronts at each generation for the second model.
    """
    # Calculate the worst point considering both series
    approx_ideal = np.min([F1.min(axis=0), F2.min(axis=0)], axis=0)
    approx_nadir = np.max([F1.max(axis=0), F2.max(axis=0)], axis=0)
    worst_point = approx_nadir
    ref_point = worst_point + (worst_point - approx_ideal) * 0.1

    color_1 = "#f44336"  # Red
    color_2 = "#008080"  # Teal

    # Calculate hypervolume for both models
    hv1 = [
        Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            ideal=F1.min(axis=0),
            nadir=F1.max(axis=0),
        ).do(_F)
        for _F in hist_F1
    ]
    hv2 = [
        Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            ideal=F2.min(axis=0),
            nadir=F2.max(axis=0),
        ).do(_F)
        for _F in hist_F2
    ]

    # Plot hypervolume convergence
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv1, color=color_1, lw=1, alpha=0.5, label=F1_name)
    plt.plot(n_evals, hv2, color=color_2, lw=1, alpha=0.5, label=F2_name)
    plt.scatter(
        n_evals,
        hv1,
        edgecolor="none",
        facecolor=color_1,
        marker="o",
        s=3,
        alpha=0.8,
    )
    plt.scatter(
        n_evals,
        hv2,
        edgecolor="none",
        facecolor=color_2,
        marker="o",
        s=3,
        alpha=0.8,
    )

    # Set x-axis ticks and labels
    plt.title("Hypervolume Convergence")
    plt.xlabel("Function Evaluations")

    # Set x-axis ticks and labels
    plt.xticks(np.arange(200000, n_evals[-1] + 1, 200000),
               [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)])
    plt.xlabel("Function Evaluations (K)")

    plt.ylabel("Hypervolume (HV)")
    plt.legend()
    plt.show()


def plot_inverted_generational_distance(IGD, n_evals, F, hist_F):
    """
    Calculates and plots the Inverted Generational Distance (IGD) convergence.

    Args:
        n_evals (np.array): The number of function evaluations at each generation.
        F (np.array): The final Pareto front.
        hist_F (list): A list of Pareto fronts at each generation.
    """

    metric = IGD(F, zero_to_one=True)
    igd = [metric.do(_F) for _F in hist_F]

    plt.plot(n_evals, igd, color='gray', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, igd, facecolor="none", edgecolor='blue', marker="o", s=5, alpha=0.7)
    plt.axhline(10 ** -2, color="red", label="10^-2", linestyle='dotted')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_inverted_generational_distance_compare(IGD, n_evals, F1, hist_F1, F2, hist_F2):
    """
    Calculates and plots the Inverted Generational Distance (IGD) convergence for two Pareto front sets.

    Args:
        IGD (pymoo.indicators.igd.IGD): The IGD class.
        n_evals (np.array): The number of function evaluations at each generation.
        F1 (np.array): The final Pareto front for the first model.
        hist_F1 (list): A list of Pareto fronts at each generation for the first model.
        F2 (np.array): The final Pareto front for the second model.
        hist_F2 (list): A list of Pareto fronts at each generation for the second model.
    """

    # Calculate IGD for both models
    igd1 = [IGD(F1, zero_to_one=True).do(_F) for _F in hist_F1]
    igd2 = [IGD(F2, zero_to_one=True).do(_F) for _F in hist_F2]

    # Plot IGD convergence
    plt.plot(n_evals, igd1, color="crimson", lw=0.7, label="Model 1")
    plt.plot(n_evals, igd2, color="blue", lw=0.7, label="Model 2")
    plt.scatter(
        n_evals,
        igd1,
        facecolor="none",
        edgecolor="crimson",
        marker="o",
        s=5,
        alpha=0.7,
    )
    plt.scatter(
        n_evals,
        igd2,
        facecolor="none",
        edgecolor="blue",
        marker="o",
        s=5,
        alpha=0.7,
    )
    plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="dotted")
    plt.title("IGD Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_inverted_generational_distance_plus(IGDPlus, n_evals, F, hist_F):
    """
    Calculates and plots the Inverted Generational Distance Plus (IGD+) convergence.

    Args:
        n_evals (np.array): The number of function evaluations at each generation.
        F (np.array): The final Pareto front.
        hist_F (list): A list of Pareto fronts at each generation.
    """

    metric = IGDPlus(F, zero_to_one=True)
    igd_plus = [metric.do(_F) for _F in hist_F]

    plt.plot(n_evals, igd_plus, color='gray', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, igd_plus, facecolor="none", edgecolor='blue', marker="o", s=5, alpha=0.7)
    plt.axhline(10 ** -2, color="red", label="10^-2", linestyle='dotted')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD+")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_inverted_generational_distance_plus_compare(IGDPlus, n_evals, F1, hist_F1, F2, hist_F2):
    """
    Calculates and plots the Inverted Generational Distance (IGDPlus) convergence for two Pareto front sets.

    Args:
        IGDPlus (pymoo.indicators.igd.IGD): The IGD class.
        n_evals (np.array): The number of function evaluations at each generation.
        F1 (np.array): The final Pareto front for the first model.
        hist_F1 (list): A list of Pareto fronts at each generation for the first model.
        F2 (np.array): The final Pareto front for the second model.
        hist_F2 (list): A list of Pareto fronts at each generation for the second model.
    """

    color_1 = "#008080"  # Teal
    color_2 = "#f44336"  # Red

    # Calculate IGD for both models
    igdplus_1 = [IGDPlus(F1, zero_to_one=True).do(_F) for _F in hist_F1]
    igdplus_2 = [IGDPlus(F2, zero_to_one=True).do(_F) for _F in hist_F2]

    export_data(igdplus_1,  "igdplus_1")
    export_data(igdplus_2,  "igdplus_2")

    # Plot IGD convergence
    plt.plot(n_evals, igdplus_1, color=color_1, lw=0.7, label="NSGA-II")
    plt.plot(n_evals, igdplus_2, color=color_2, lw=0.7, label="NSGA-III")
    plt.scatter(
        n_evals,
        igdplus_1,
        facecolor="none",
        edgecolor=color_1,
        marker="o",
        s=5,
        alpha=0.7,
    )
    plt.scatter(
        n_evals,
        igdplus_2,
        facecolor="none",
        edgecolor=color_2,
        marker="o",
        s=5,
        alpha=0.7,
    )
    # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="dotted")
    plt.title("IGD+ Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD+")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_inverted_generational_distance_plus_compare_inputigdplus(n_evals, sce_label_1, sce_label_2):
    """
    Plots the Inverted Generational Distance Plus (IGD+) convergence for two algorithms.

    Args:
        n_evals (np.array): Function evaluations at each generation.
        sce_label_1 (str): Label for the first scenario (e.g., "NSGA-II").
        sce_label_2 (str): Label for the second scenario (e.g., "NSGA-III").
    """

    color_1 = "#008080"  # Teal for NSGA-II
    color_2 = "#f44336"  # Red for NSGA-III

    faded_color_1 = "#80B2B2"
    faded_color_2 = "#F8A096"
    # Load IGD+ data
    igdplus_1 = get_data("EXP_igdplus_1")
    igdplus_2 = get_data("EXP_igdplus_2")

    # Validate data lengths
    if len(n_evals) != len(igdplus_1) or len(n_evals) != len(igdplus_2):
        raise ValueError("Mismatch in data lengths: n_evals and IGD+ values must have the same length.")

    plt.figure(figsize=(7, 5))

    # Plot actual lines with alpha transparency
    plt.plot(n_evals, igdplus_1, color=color_1, lw=2, alpha=0.9)
    plt.plot(n_evals, igdplus_2, color=color_2, lw=2, alpha=0.9)

    # Scatter points
    # plt.scatter(n_evals, igdplus_1, facecolor=color_1, edgecolors="none", marker="o", s=4, alpha=0.2)
    # plt.scatter(n_evals, igdplus_2, facecolor=color_2, edgecolors="none", marker="o", s=4, alpha=0.2)

    # Create invisible lines for a separate legend (full opacity)
    legend_line1, = plt.plot([], [], color=color_1, lw=2, label=sce_label_1)
    legend_line2, = plt.plot([], [], color=color_2, lw=2, label=sce_label_2)

    # Title and labels
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD+")

    # X-ticks formatting
    plt.xticks(
        np.arange(200000, n_evals[-1] + 1, 200000),
        [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)]
    )

    # Log scale for better IGD+ visualization
    plt.yscale("log")

    # Grid settings
    plt.grid(True, linestyle="dotted", alpha=0.5)

    # Add separate legend (with opaque lines)
    plt.legend(handles=[legend_line1, legend_line2], fontsize=11, frameon=True, loc="best")

    plt.show()


def plot_running_igd_compare(n_evals, F1, hist_F1, F1_name, F2, hist_F2, F2_name, window_size=10):
    """
    Plots the running IGD metric of two models for comparison.

    Args:
        n_evals (list): List of function evaluation counts.
        F1 (np.array): Final approximation set of the first model.
        hist_F1 (list): List of approximation sets of the first model at each generation.
        F1_name (str): Name of the first model.
        F2 (np.array): Final approximation set of the second model.
        hist_F2 (list): List of approximation sets of the second model at each generation.
        F2_name (str): Name of the second model.
        window_size (int): Size of the sliding window for the running metric.
    """

    # Initialize the running metric values for both models
    running_igd1 = []
    running_igd2 = []

    # Determine the length of the shorter history
    min_len = min(len(hist_F1), len(hist_F2))

    # Iterate through the history of approximation sets
    for i in range(min_len):
        # Determine the start and end indices of the window
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1

        # Extract the approximation sets within the window for both models
        window_F1 = hist_F1[start_idx:end_idx]
        window_F2 = hist_F2[start_idx:end_idx]

        # Calculate the ideal and nadir points based on the combined window
        combined_F = np.vstack(window_F1 + window_F2)
        approx_ideal = np.min(combined_F, axis=0)
        approx_nadir = np.max(combined_F, axis=0)

        # Normalize the current approximation sets for both models
        epsilon = 1e-10
        normalized_F1 = (hist_F1[i] - approx_ideal) / (approx_nadir - approx_ideal + epsilon)
        normalized_F2 = (hist_F2[i] - approx_ideal) / (approx_nadir - approx_ideal + epsilon)
        # Calculate the IGD for the current approximation sets
        igd1 = IGD(normalized_F1).do(normalized_F1)
        igd2 = IGD(normalized_F2).do(normalized_F2)

        # Append the IGD values to the running metric values
        running_igd1.append(igd1)
        running_igd2.append(igd2)

    # Plot the running metric values for both models
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals[:min_len], running_igd1, color="#008080", lw=1, alpha=0.5, label=F1_name)
    plt.plot(n_evals[:min_len], running_igd2, color="#f44336", lw=1, alpha=0.5, label=F2_name)
    plt.scatter(n_evals[:min_len], running_igd1, edgecolor="none", facecolor="#008080", marker="o", s=3, alpha=0.8)
    plt.scatter(n_evals[:min_len], running_igd2, edgecolor="none", facecolor="#f44336", marker="o", s=3, alpha=0.8)

    plt.title("Running IGD Convergence Comparison")
    plt.xlabel("Function Evaluations (K)")
    plt.xticks(np.arange(200000, n_evals[-1] + 1, 200000),
               [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)])
    plt.ylabel("Running IGD")
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pymoo.indicators.rmetric import RMetric
from pymoo.indicators.igd import IGD
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort


def compute_running_igd(n_evals, hist_F, window_size=10000, save_path="running_igd.csv"):
    """
    Computes the running IGD metric using pymoo's RunningMetric.
    If the data is already saved, loads it instead of recalculating.

    Args:
        n_evals (list): List of function evaluation counts.
        hist_F (list): List of approximation sets at each generation.
        window_size (int): Size of the sliding window for the running metric.
        save_path (str): File path to save/load the computed running IGD.

    Returns:
        list: Running IGD values.
    """
    if os.path.exists(save_path):
        print("Loading precomputed running IGD data...")
        df = pd.read_csv(save_path)
        return df["running_igd"].tolist()

    # Combine all generations' solutions to approximate the Pareto front
    sample_size = min(10000, sum(len(F) for F in hist_F))
    indices = np.random.choice(sum(len(F) for F in hist_F), sample_size, replace=False)
    combined_F = np.vstack(hist_F)[indices]
    fronts = fast_non_dominated_sort(combined_F, n_stop_if_ranked=1)
    true_pf = combined_F[fronts[0]]  # Use only the first non-dominated front

    igd_metric = IGD(true_pf)
    running_igd = []
    total = len(hist_F)


    for i, F in enumerate(hist_F):
        print(f"{i + 1}/{total} calculating...")
        running_igd.append(igd_metric.do(F))

    # Save computed running IGD data to CSV
    df = pd.DataFrame({"n_evals": n_evals[:len(running_igd)], "running_igd": running_igd})
    df.to_csv(save_path, index=False)

    # Save numeric IGD metrics
    with open(f"{save_path.replace('.csv', '_values.txt')}", 'w') as f:
        f.write(f"Final Running IGD: {running_igd[-1]}")
        f.write(f"Average Running IGD: {np.mean(running_igd)}")
        f.write(f"Minimum Running IGD: {np.min(running_igd)}")
    print(f"Final Running IGD: {running_igd[-1]}")
    print(f"Average Running IGD: {np.mean(running_igd)}")
    print(f"Minimum Running IGD: {np.min(running_igd)}")

    return running_igd


def export_running_igd(n_evals, hist_F, window_size=10000, save_path="running_igd.csv"):
    """
    Computes and exports the running IGD metric to a file.

    Args:
        n_evals (list): List of function evaluation counts.
        hist_F (list): List of approximation sets at each generation.
        window_size (int): Size of the sliding window.
        save_path (str): File path to save the computed running IGD.
    """
    running_igd = compute_running_igd(n_evals, hist_F, window_size, save_path)
    print(f"Running IGD data saved to {save_path}")


def plot_running_igd(n_evals, hist_F, model_name, window_size=10000, save_path="running_igd.csv"):
    """
    Plots the running IGD metric for a single model.

    Args:
        n_evals (list): List of function evaluation counts.
        hist_F (list): List of approximation sets at each generation.
        model_name (str): Name of the model.
        window_size (int): Size of the sliding window.
        save_path (str): File path to load/save the computed running IGD.
    """
    running_igd = compute_running_igd(n_evals, hist_F, window_size, save_path)
    min_len = len(running_igd)

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals[:min_len], running_igd, color="#008080", lw=1, alpha=0.7, label=f"{model_name} (Running IGD)")
    plt.scatter(n_evals[:min_len], running_igd, edgecolor="none", facecolor="#008080", marker="o", s=3, alpha=0.8)

    plt.yscale("log")  # Ensure logarithmic scale for better visualization
    plt.title("Running IGD Convergence")
    plt.xlabel("Function Evaluations (K)")
    plt.xticks(np.arange(200000, n_evals[-1] + 1, 200000),
               [f"{int(x / 1000)}K" for x in np.arange(200000, n_evals[-1] + 1, 200000)])
    plt.ylabel("Running IGD")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def plot_pareto_3d(RES_F):
    # Extract and normalize data
    res_x = RES_F[:, 0]
    res_y = RES_F[:, 1]
    res_z = RES_F[:, 2]

    res_x_norm = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    res_y_norm = (res_y - np.min(res_y)) / (np.max(res_y) - np.min(res_y))
    res_z_norm = (res_z - np.min(res_z)) / (np.max(res_z) - np.min(res_z))

    # Compute average for coloring
    c_value = np.mean([res_x_norm, res_y_norm, res_z_norm], axis=0)

    # Define boundaries for 5 discrete sections
    def create_boundaries(data, num_sections=5):
        # Calculate the percentiles for the given number of sections
        percentiles = np.linspace(0, 100, num=num_sections + 1)
        boundaries = np.percentile(data, percentiles)
        return boundaries

    # Get boundaries for coloring based on percentiles
    boundaries = create_boundaries(c_value, num_sections=5)

    # Assign colors based on boundaries
    color_indices = np.digitize(c_value, boundaries) - 1
    my_cmap = plt.get_cmap('RdYlGn_r', len(boundaries) - 1)
    colors = my_cmap(color_indices / (len(boundaries) - 2))

    # Plotting
    fig = plt.figure(figsize=(16, 9))
    ax3d = fig.add_subplot(111, projection='3d')

    # Scatter plot with discrete color sections
    scatter = ax3d.scatter3D(
        res_z, res_y, res_x, alpha=0.8, c=colors, marker='o', s=30
    )

    # Labels and title
    ax3d.set_title('NSGA PARETO', fontweight='bold')
    ax3d.set_xlabel('Deviation from Natural Flow', fontweight='bold')
    ax3d.set_ylabel('Irrigation Deficits', fontweight='bold')
    ax3d.set_zlabel('Deficit from the Potential Production', fontweight='bold')

    # Formatting
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.set_major_locator(FixedLocator(axis.get_ticklocs()))
        axis.set_tick_params(pad=5)

    # Custom tick labels
    ax3d.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax3d.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax3d.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f"{z:,.3f}"))

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.4, label='Score')
    # cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.tick_params(pad=10)

    # # 2D projections
    # min_x, max_x = np.min(res_x), np.max(res_x)
    # min_y, max_y = np.min(res_y), np.max(res_y)
    # min_z, max_z = np.min(res_z), np.max(res_z)

    # ax3d.scatter(res_z, res_y, min_x, alpha=0.1, c='grey', marker='o')
    # ax3d.scatter(res_z, max_y, res_x, alpha=0.1, c='grey', marker='o')
    # ax3d.scatter(min_z, res_y, res_x, alpha=0.1, c='grey', marker='o')

    # plt.tight_layout()
    plt.show()


def plot_pareto_2d(RES_F):
    res_x = RES_F[:, 0]
    res_y = RES_F[:, 1]
    res_z = RES_F[:, 2]

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.suptitle("NSGA-III PARETO", fontsize=16, fontweight="bold")
    my_cmap = plt.get_cmap("RdYlGn_r")  # Use reversed RdYlGn for green to red

    # Define boundaries for 5 discrete sections
    def create_boundaries(data, num_sections=5):
        # # Equally
        # min_val, max_val = np.min(data), np.max(data)
        # boundaries = np.linspace(min_val, max_val, num=num_sections + 1)

        # Calculate the percentiles for the given number of sections
        percentiles = np.linspace(0, 100, num=num_sections + 1)
        boundaries = np.percentile(data, percentiles)

        return boundaries

    # Scatter plot 1: Energy vs Irrigation
    ax = fig.add_subplot(311, adjustable="box")
    ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    boundaries_z = create_boundaries(res_z)
    norm_z = BoundaryNorm(boundaries_z, my_cmap.N, clip=True)
    sctt = ax.scatter(res_x, res_y, c=res_z, s=10, alpha=0.8, marker="o", cmap=my_cmap, norm=norm_z)
    ax.set_xlabel("Energy", fontweight="bold")
    ax.set_ylabel("Irrigation", fontweight="bold")
    cbar = fig.colorbar(sctt, ax=ax, boundaries=boundaries_z, label="Ecology", format='%1.1f')
    cbar.ax.yaxis.label.set_fontweight("bold")

    # Scatter plot 2: Irrigation vs Ecology
    ax = fig.add_subplot(312, adjustable="box")
    ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    boundaries_x = create_boundaries(res_x)
    norm_x = BoundaryNorm(boundaries_x, my_cmap.N, clip=True)
    sctt = ax.scatter(res_y, res_z, c=res_x, s=10, alpha=0.8, marker="o", cmap=my_cmap, norm=norm_x)
    ax.set_xlabel("Irrigation", fontweight="bold")
    ax.set_ylabel("Ecology", fontweight="bold")
    cbar = fig.colorbar(sctt, ax=ax, boundaries=boundaries_x, label="Energy", format='%1.1f')
    cbar.ax.yaxis.label.set_fontweight("bold")

    # Scatter plot 3: Energy vs Ecology
    ax = fig.add_subplot(313, adjustable="box")
    ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    boundaries_y = create_boundaries(res_y)
    norm_y = BoundaryNorm(boundaries_y, my_cmap.N, clip=True)
    sctt = ax.scatter(res_x, res_z, c=res_y, s=10, alpha=0.8, marker="o", cmap=my_cmap, norm=norm_y)
    ax.set_xlabel("Energy", fontweight="bold")
    ax.set_ylabel("Ecology", fontweight="bold")
    cbar = fig.colorbar(sctt, ax=ax, boundaries=boundaries_y, label="Irrigation", format='%1.1f')
    cbar.ax.yaxis.label.set_fontweight("bold")

    plt.show()


def plot_pareto_2obj(RES_F):
    res_x = RES_F[:, 0]
    res_y = RES_F[:, 1]

    res_x_norm = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    res_y_norm = (res_y - np.min(res_y)) / (np.max(res_y) - np.min(res_y))

    average_matrix = np.mean([res_x_norm, res_y_norm], axis=0)
    c_value = average_matrix

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("NSGA PARETO FRONT", fontsize=14, fontweight="bold")
    my_cmap = plt.get_cmap("jet_r")

    ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)

    sctt = ax.scatter(res_x, res_y, c=c_value, s=10, alpha=0.8, marker="o", cmap=my_cmap)
    ax.set_xlabel("Objective 1", fontweight="bold")
    ax.set_ylabel("Objective 2", fontweight="bold")
    fig.colorbar(sctt, ax=ax, label="Score")

    plt.show()


def plot_obj_total(RES_COUNT, GEN_WATER, EN_DEMAND, IRG_WATER, IR_DEMAND, OUTFLOW, OBS_OUTFLOW, OBS_NAT_FLOW):
    # Setting a smaller font size for all subplots
    font_size = 8

    title = get_data_string("CH_NAMES")

    sim_en_tot = np.sum(GEN_WATER, axis=1)
    obs_en_tot = np.sum(EN_DEMAND, axis=1)
    sim_ir_tot = np.sum(IRG_WATER, axis=1)
    obs_ir_tot = np.sum(IR_DEMAND, axis=1)
    sim_flow_tot = np.sum(OUTFLOW, axis=1)
    obs_flow_tot = np.sum(OBS_OUTFLOW, axis=1)
    obs_natflow_tot = np.sum(OBS_NAT_FLOW, axis=1)
    tot_exen = ["-", "OBS", "SIM", "-"]
    tot_exen_2 = ["-", "OBS", "NAT", "SIM", "-"]

    fig, axs = plt.subplots(3, RES_COUNT + 1, figsize=(16, 9))
    fig.suptitle("AMAC FONK. TOPLAMLARI (ENERJI-SULAMA-EKOLJI)", fontsize=16, fontweight="bold")

    # Plotting OBJ 1 - Total Energy
    for qq in range(RES_COUNT):
        axs[0, qq].bar(tot_exen[1], obs_en_tot[qq], color="gold", alpha=0.9, label="OBS ENERJI")
        axs[0, qq].bar(tot_exen[2], sim_en_tot[qq], color="tab:red", alpha=0.9, label="SIM ENERJI")
        axs[0, qq].set_title(title[qq], fontsize=font_size)
        axs[0, qq].grid(color="grey", linestyle=":", linewidth=0.5)
        axs[0, qq].tick_params(axis='both', which='major', labelsize=font_size)

    # Plotting total energy for all reservoirs
    axs[0, -1].set_title(title[-1], fontweight="bold", fontsize=font_size)
    axs[0, -1].set_facecolor("snow")
    axs[0, -1].bar(tot_exen[1], np.sum(obs_en_tot), color="gold", label="OBS TOT ENERJI")
    axs[0, -1].bar(tot_exen[2], np.sum(sim_en_tot), color="tab:red", label="SIM TOT ENERJI")
    axs[0, -1].grid(color="grey", linestyle=":", linewidth=0.5)
    axs[0, -1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=font_size)

    # Plotting OBJ 2 - Total Irrigation
    for qq in range(RES_COUNT):
        axs[1, qq].bar(tot_exen[1], obs_ir_tot[qq], color="tab:green", alpha=0.9, label="OBS SULAMA")
        axs[1, qq].bar(tot_exen[2], sim_ir_tot[qq], color="tab:red", alpha=0.9, label="SIM SULAMA")
        axs[1, qq].grid(color="grey", linestyle=":", linewidth=0.5)
        axs[1, qq].tick_params(axis='both', which='major', labelsize=font_size)

    axs[1, -1].set_facecolor("snow")
    axs[1, -1].bar(tot_exen[1], np.sum(obs_ir_tot), color="tab:green", label="OBS TOT SULAMA")
    axs[1, -1].bar(tot_exen[2], np.sum(sim_ir_tot), color="tab:red", label="SIM TOT SULAMA")
    axs[1, -1].grid(color="grey", linestyle=":", linewidth=0.5)
    axs[1, -1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=font_size)

    # Plotting OBJ 3 - Total Water Release from Dam
    for qq in range(RES_COUNT):
        axs[2, qq].bar(tot_exen_2[1], obs_flow_tot[qq], color="darkviolet", alpha=0.9, label="OBS TOT DAM OUT")
        axs[2, qq].bar(tot_exen_2[2], obs_natflow_tot[qq], color="lightseagreen", alpha=0.9, label="OBS TOT NAT FLOW")
        axs[2, qq].bar(tot_exen_2[3], sim_flow_tot[qq], color="tab:red", alpha=0.9, label="OBS SIM DAM OUT")
        axs[2, qq].grid(color="grey", linestyle=":", linewidth=0.5)
        axs[2, qq].tick_params(axis='both', which='major', labelsize=font_size)

    axs[2, -1].set_facecolor("snow")
    axs[2, -1].bar(tot_exen_2[1], obs_flow_tot[4], color="darkviolet", label="OBS TOT DAM OUT")
    axs[2, -1].bar(tot_exen_2[2], obs_natflow_tot[4], color="lightseagreen", label="OBS TOT NAT_FLOW")
    axs[2, -1].bar(tot_exen_2[3], sim_flow_tot[4], color="tab:red", label="SIM DAM OUT")
    axs[2, -1].grid(color="grey", linestyle=":", linewidth=0.5)
    axs[2, -1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=font_size)

    plt.tight_layout()
    # plt.show()


def plot_objectives(RES_COUNT, GEN_WATER, IRG_WATER, EN_DEMAND, IR_DEMAND, OUTFLOW, OBS_NAT_FLOW, UPPER_BOUND_CSV, LOWER_BOUND_CSV, OBS_EN_MIN,
                    OBS_EN_MAX):
    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")
    tot_exen = ["-", "OBS", "SIM", "-"]
    tot_exen_2 = ["-", "OBS", "NAT", "SIM", "-"]

    # Setting a smaller font size for all subplots
    font_size = 8
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.suptitle(
        "AMAC FONKSIYONLARI (ENERJI-SULAMA-EKOLOJI)", fontsize=16, fontweight="bold"
    )

    for qq in range(RES_COUNT):
        ax = fig.add_subplot(3, RES_COUNT, qq + 1)
        sim_en_gr = GEN_WATER[qq, :]
        obs_en_gr = EN_DEMAND[qq, :]
        obs_en_min = OBS_EN_MIN[qq, :]
        obs_en_max = OBS_EN_MAX[qq, :]
        ax.plot(months, obs_en_gr, color="gold", label="OBS ENERJI")
        ax.fill_between(months, obs_en_max, obs_en_min, color="gold", alpha=0.2)
        ax.plot(months, sim_en_gr, color="tab:red", label="SIM ENERJI")
        ax.set_title(title[qq], fontsize=font_size)
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

    for qq in range(RES_COUNT):
        ax = fig.add_subplot(3, RES_COUNT, RES_COUNT + qq + 1)
        sim_ir_gr = IRG_WATER[qq, :]
        obs_ir_gr = IR_DEMAND[qq, :]
        # obs_ir_min = LOWER_BOUND_CSV[qq + RES_COUNT, :]
        # obs_ir_max = UPPER_BOUND_CSV[qq + RES_COUNT, :]
        # ax.fill_between(months, obs_ir_max, obs_ir_min, color="tab:green", alpha=0.2)
        # ax.plot(months, obs_ir_gr, color="tab:green", label="OBS SULAMA")
        # ax.plot(months, sim_ir_gr, color="tab:red", label="SIM SULAMA")
        # plt.grid(color="grey", linestyle=":", linewidth=0.5)
        # ax.tick_params(axis='both', which='major', labelsize=font_size)

    for qq in range(RES_COUNT):
        ax = fig.add_subplot(3, RES_COUNT, 2 * RES_COUNT + qq + 1)
        sim_s = OUTFLOW[qq, :]
        obs_s = OBS_OUTFLOW[qq, :]
        # nat_s = OBS_NAT_FLOW[qq, :]
        nat_s = [226.058, 280.205, 427.726, 494.225, 501.786, 614.908, 565.886, 422.009, 332.352, 313.05, 271.222,
                 201.754]
        # obs_damout_min = OBS_NAT_FLOW[qq, :] * 0.9
        # obs_damout_max = OBS_NAT_FLOW[qq, :] * 1.1
        # ax.fill_between(
        #     months, obs_damout_max, obs_damout_min, color="lightseagreen", alpha=0.2
        # )
        ax.plot(months, obs_s, color="darkviolet", label="OBS DAM OUT")
        ax.plot(months, nat_s, color="lightseagreen", label="NAT FLOW")
        ax.plot(months, sim_s, color="tab:red", label="SIM DAM OUT")
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()
    # plt.show()


def plot_energy_sector(GEN_EN_GWH, OBS_ENR_GWH, GEN_WATER, EN_DEMAND, POWER_MW, OBS_EN_FIRM, OBS_EN_FIRM_MW, OBS_EN_FIRM_GWH):
    ENG_RES_COUNT = 5
    ENG_NODES = [4, 5, 7, 8, 9]

    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")

    EN_SECN = np.maximum(GEN_WATER - OBS_EN_FIRM, 0)
    EN_FIRM = np.maximum(OBS_EN_FIRM, GEN_WATER)
    EN_SECN_MW = np.maximum(POWER_MW - OBS_EN_FIRM_MW, 0)
    EN_FIRM_MW = np.maximum(OBS_EN_FIRM_MW, POWER_MW)
    EN_SECN_GWH = np.maximum(GEN_EN_GWH - OBS_EN_FIRM_GWH, 0)
    EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, GEN_EN_GWH)

    blue_main = "#3A5775"  # pale blue
    blue_shade = "#8CA2B0"  # light blue
    orange_main = "#D95B16"  # pale orange
    orange_shade = "#FFA144"  # light orange
    crimson_main = "#692139"  # pale crimson
    crimson_shade = "#B7505F"  # light crimson

    fig, axs = plt.subplots(3, ENG_RES_COUNT, figsize=(16, 9), sharey="row")
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.suptitle("Hydropower Energy", fontsize=16, fontweight="bold")

    for i, node in enumerate(ENG_NODES):
        idx = node - 1

        ax = axs[0, i]
        ax.bar(months, EN_FIRM[idx, :], color=blue_main, label="Firm Energy (HM3)")
        ax.bar(months, EN_SECN[idx, :], bottom=EN_FIRM[idx, :], color=blue_shade, label="Secondary Energy (HM3)")
        ax.scatter(months, EN_DEMAND[idx, :], color="grey", edgecolor="silver", label="Observation", s=15, alpha=1, zorder=4)
        ax.set_title(title[idx])
        ax.set_axisbelow(True)
        ax.grid(color="grey", linestyle=":", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)

        ax = axs[1, i]
        ax.bar(months, EN_FIRM_MW[idx, :], color=orange_main, label="Firm Energy (MW)")
        ax.bar(months, EN_SECN_MW[idx, :], bottom=EN_FIRM_MW[idx, :], color=orange_shade, label="Secondary Energy (MW)")
        ax.scatter(months, OBS_POW_MW[idx, :], color="grey", edgecolor="silver", label="Observation", s=15, alpha=1, zorder=4)
        ax.set_axisbelow(True)
        ax.grid(color="grey", linestyle=":", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)

        ax = axs[2, i]
        ax.bar(months, EN_FIRM_GWH[idx, :], color=crimson_main, label="Firm Energy (GWH)")
        ax.bar(months, EN_SECN_GWH[idx, :], bottom=EN_FIRM_GWH[idx, :], color=crimson_shade, label="Secondary Energy (GWH)")
        ax.scatter(months, OBS_EN_GWH[idx, :], color="grey", edgecolor="silver", label="Observation", s=15, alpha=1, zorder=4)
        ax.set_xlabel("Months")
        ax.set_axisbelow(True)
        ax.grid(color="grey", linestyle=":", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Set labels and legend
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            if i == 2:  # Set x-label for the bottom row of plots
                ax.set_xlabel("Months")
            if j == ENG_RES_COUNT:  # Add legend to the last plot of each row
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            if i == 0 and j == 0:  # Set ylabel for only the first plot in the first row
                ax.set_ylabel("Water for Energy Generation (hm³)")
            elif i == 1 and j == 0:  # Set ylabel for only the first plot in the second row
                ax.set_ylabel("Total Power (MW)")
            elif i == 2 and j == 0:  # Set ylabel for only the first plot in the third row
                ax.set_ylabel("Generated Energy (GWh)")

    plt.tight_layout()
    plt.show()


def plot_boxplot(SIM_OUTPUT, OBS_DATA, NODE_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None, debug_mode=False, debug_idx=0):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)

    for node_idx, node_name in enumerate(NODE_NAMES):
        node_data = SIM_OUTPUT[:, node_idx, :]  # Extract data for the current reservoir

        if debug_mode and node_idx != debug_idx:
            continue  # Skip plotting if debugging single reservoir and current reservoir is not the one to plot

        # Create boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(node_data, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
                    flierprops=flierprops, medianprops=medianprops, meanprops=meanprops,
                    patch_artist=True, widths=0.5)

        # Plot observed mean
        obs = OBS_DATA[node_idx, :]
        plt.scatter(np.arange(1, 13), obs, color="c", edgecolor='teal', marker='o', alpha=0.9, s=50, label="Observed Mean", zorder=3)
        smo_obs = gaussian_filter1d(obs, sigma=0.3)
        plt.plot(np.arange(1, 13), smo_obs, color="c", linestyle='-', alpha=0.3, linewidth=3, zorder=3)

        # Plot simulated mean
        sim = np.mean(node_data, axis=0)
        plt.scatter(np.arange(1, 13), sim, color="crimson", edgecolor='maroon', marker='o', alpha=0.9, s=50, label="Simulated Mean", zorder=4)
        smo_sim = gaussian_filter1d(sim, sigma=0.3)
        plt.plot(np.arange(1, 13), smo_sim, color="crimson", linestyle='-', alpha=0.3, linewidth=3, zorder=4)

        # Simulated Distribution
        for month_idx in range(12):
            month_val = node_data[:, month_idx]
            x_val = np.repeat(month_idx + 1, len(month_val))
            plt.scatter(x_val, month_val, edgecolor='gainsboro', alpha=0.1, color='none')

        # Limits
        if LOWER_LIMIT is not None and UPPER_LIMIT is not None and INITIAL is not None:
            plt.axhline(UPPER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            plt.axhline(LOWER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            plt.axhline(INITIAL[node_idx], color="khaki", linestyle="-", linewidth=2, alpha=0.5, zorder=1)

            # Get the upper and lower limits of the data
            data_min = min(LOWER_LIMIT[node_idx])
            data_max = max(UPPER_LIMIT[node_idx])
            plot_lower_limit, plot_upper_limit = plt.gca().get_ylim()
            lower_limit = min(data_min, plot_lower_limit)
            upper_limit = max(data_max, plot_upper_limit)
            plt.axhspan(lower_limit, LOWER_LIMIT[node_idx][0], facecolor='lightgrey', alpha=0.3)
            plt.axhspan(UPPER_LIMIT[node_idx][0], upper_limit, facecolor='lightgrey', alpha=0.3)
            plt.gca().set_ylim(lower_limit, upper_limit)

        plt.title(f'{node_name}')
        plt.xlabel('Months')
        plt.ylabel('Reservoir Elevation (m)')
        plt.xticks(ticks=np.arange(1, 13), labels=CH_MONTHS_NAME)
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()


def plot_boxplot_collect(SIM_OUTPUT, OBS_DATA, NODE_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)

    num_plots = len(NODE_NAMES)
    fig, axs = plt.subplots(1, num_plots, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    for ax, node_name in zip(axs, NODE_NAMES):
        node_idx = np.where(NODE_NAMES == node_name)[0][0]  # Index of the current reservoir
        node_data = SIM_OUTPUT[:, node_idx, :]  # Extract data for the current reservoir

        # Create boxplot
        ax.boxplot(node_data, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
                   flierprops=flierprops, medianprops=medianprops, meanprops=meanprops,
                   patch_artist=True, widths=0.5)

        # Plot observed mean
        obs = OBS_DATA[node_idx, :]
        ax.scatter(np.arange(1, 13), obs, color="c", edgecolor='teal', marker='o', alpha=0.9, s=50, label="Observed Mean", zorder=3)
        smo_obs = gaussian_filter1d(obs, sigma=0.3)
        ax.plot(np.arange(1, 13), smo_obs, color="c", linestyle='-', alpha=0.3, linewidth=3, zorder=3)

        # Plot simulated mean
        sim = np.mean(node_data, axis=0)
        ax.scatter(np.arange(1, 13), sim, color="crimson", edgecolor='maroon', marker='o', alpha=0.9, s=50, label="Simulated Mean", zorder=4)
        smo_sim = gaussian_filter1d(sim, sigma=0.3)
        ax.plot(np.arange(1, 13), smo_sim, color="crimson", linestyle='-', alpha=0.3, linewidth=3, zorder=4)

        # Simulated Distribution
        for month_idx in range(12):
            month_val = node_data[:, month_idx]
            x_val = np.repeat(month_idx + 1, len(month_val))
            ax.scatter(x_val, month_val, edgecolor='gainsboro', alpha=0.1, color='none')

        # Limits as lines instead of area
        if LOWER_LIMIT is not None and UPPER_LIMIT is not None and INITIAL is not None:
            ax.plot(np.arange(1, 13), UPPER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2, label="Upper Limit")
            ax.plot(np.arange(1, 13), LOWER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2, label="Lower Limit")
            ax.plot(np.arange(1, 13), INITIAL[node_idx], color="khaki", linestyle="-", linewidth=2, alpha=0.5, zorder=1, label="Initial")

        ax.set_title(f'{node_name}')
        ax.set_xlabel('Months')
        ax.set_xticks(ticks=np.arange(1, 13))
        ax.grid(True)
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def plot_scatter_collect(SIM_OUTPUT, OBS_DATA, NODE_NAMES, MONTH_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)

    num_plots = len(NODE_NAMES)
    fig, axs = plt.subplots(1, num_plots, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    for ax, node_name in zip(axs, NODE_NAMES):
        node_idx = np.where(NODE_NAMES == node_name)[0][0]  # Index of the current reservoir
        node_data = SIM_OUTPUT[:, node_idx, :]  # Extract data for the current reservoir

        # # Create boxplot
        # ax.boxplot(node_data, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
        #            flierprops=flierprops, medianprops=medianprops, meanprops=meanprops,
        #            patch_artist=True, widths=0.5)

        # Plot observed mean
        obs = OBS_DATA[node_idx, :]
        ax.scatter(np.arange(1, 13), obs, color="c", edgecolor='teal', marker='o', alpha=0.9, s=50, label="Observed Mean", zorder=3)
        smo_obs = gaussian_filter1d(obs, sigma=0.3)
        ax.plot(np.arange(1, 13), smo_obs, color="c", linestyle='-', alpha=0.3, linewidth=3, zorder=3)

        # Plot simulated mean
        sim = np.mean(node_data, axis=0)
        ax.scatter(np.arange(1, 13), sim, color="crimson", edgecolor='maroon', marker='o', alpha=0.9, s=50, label="Simulated Mean", zorder=4)
        smo_sim = gaussian_filter1d(sim, sigma=0.3)
        ax.plot(np.arange(1, 13), smo_sim, color="crimson", linestyle='-', alpha=0.3, linewidth=3, zorder=4)

        # Simulated Distribution
        for month_idx in range(12):
            month_val = node_data[:, month_idx]
            x_val = np.repeat(month_idx + 1, len(month_val))
            ax.scatter(x_val, month_val, color="#DC143C", edgecolor='crimson', alpha=0.3)

        # Limits
        if LOWER_LIMIT is not None and UPPER_LIMIT is not None and INITIAL is not None:
            ax.axhline(UPPER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            ax.axhline(LOWER_LIMIT[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            ax.axhline(INITIAL[node_idx], color="khaki", linestyle="-", linewidth=2, alpha=0.5, zorder=1)

            # Get the upper and lower limits of the data
            data_min = min(LOWER_LIMIT[node_idx])
            data_max = max(UPPER_LIMIT[node_idx])
            plot_lower_limit, plot_upper_limit = ax.get_ylim()
            lower_limit = min(data_min, plot_lower_limit)
            upper_limit = max(data_max, plot_upper_limit)
            ax.axhspan(lower_limit, LOWER_LIMIT[node_idx][0], facecolor='lightgrey', alpha=0.3)
            ax.axhspan(UPPER_LIMIT[node_idx][0], upper_limit, facecolor='lightgrey', alpha=0.3)
            ax.set_ylim(lower_limit, upper_limit)

        ax.set_title(f'{node_name}')
        ax.set_xlabel('Months')
        # ax.set_ylabel('Reservoir Storage (hm³)')
        ax.set_xticks(ticks=np.arange(1, 13))
        ax.set_xticklabels(MONTH_NAMES)
        ax.grid(True)
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def adjust_data_series(data):
    return np.concatenate((data, data[:, :, :1]), axis=2)


from matplotlib.ticker import FuncFormatter


def thousands_formatter_fixed(x, pos):
    """1000 -> 1.000 formatını sağlar (Yerel ayardan bağımsız)."""
    # Sayıyı tam sayıya yuvarlar ve binlik ayıracı ekler.
    s = f'{int(round(x)):d}'

    # Binlik ayıracı ekleme (Python'ın yerleşik binlik ayıracı (,) yerine nokta (.))
    result = []
    for i, digit in enumerate(reversed(s)):
        if i > 0 and i % 3 == 0:
            result.append('.')
        result.append(digit)

    return "".join(reversed(result))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_node_individually_turkish(sim_outputs, obs_data, node_names, month_names,
                                   lower_limit=None, upper_limit=None, initial=None):
    """
    Generates an individual plot for each reservoir node, using the steps-pre
    style for the mean, shaded bands/error bars for variability, and uses
    Turkish month abbreviations for the x-axis labels.
    """

    # --- DEFINITIONS ---
    # Turkish Month Abbreviations (4 characters)
    TR_MONTHS = [ 'Eki', 'Kas', 'Ara', 'Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl']

    # Colors (based on your final aesthetic)
    PLOT_ORDER = [1, 2, 0]
    plot_colors = {
        0: {"line": "#993399", "fill": "#D0BBD0", "error": "#993399"},  # Normal (Purple)
        1: {"line": "#009999", "fill": "#B0E0E6", "error": "#009999"},  # Wet (Teal)
        2: {"line": "#FF7F00", "fill": "#FFC080", "error": "#FF7F00"},  # Dry (Orange)
    }
    MODEL_LABELS = ["Normal Su Yılı", "Islak Su Yılı", "Kuru Su Yılı"]

    # --- CONSTRAINT PROCESSING ---
    if lower_limit is not None and upper_limit is not None and initial is not None:
        const_lower_vals = lower_limit[:, 0]
        const_upper_vals = upper_limit[:, 0]
    else:
        const_lower_vals, const_upper_vals = None, None

    # --- LOOP through each node ---
    for node_idx, node_name in enumerate(node_names):

        fig, ax = plt.subplots(figsize=(8, 5))

        # --- Plot Simulation Data (Mean, Step Line, Error Bars, and Bands) ---
        for model_idx in PLOT_ORDER:
            p_color = plot_colors[model_idx]["line"]
            p_fill = plot_colors[model_idx]["fill"]

            sim_output = sim_outputs[model_idx]
            node_data = sim_output[:, node_idx, :]

            # Calculate Statistics (Mean and Standard Deviation)
            sim_mean = np.mean(node_data, axis=0)
            sim_std = np.std(node_data, axis=0)

            # Cyclic extension (13th point)
            sim_mean = np.concatenate((sim_mean, [sim_mean[0]]))
            sim_std = np.concatenate((sim_std, [sim_std[0]]))
            months = np.arange(0, 13)

            # Define bounds: Mean +/- Std Dev
            lower_bound = sim_mean - sim_std
            upper_bound = sim_mean + sim_std

            # 1. SHADED CONFIDENCE BAND (Mean +/- Std Dev)
            ax.fill_between(months, lower_bound, upper_bound,
                            color=p_fill,
                            alpha=0.5,
                            step='pre',
                            zorder=2)

            # 2. MEAN PLOTTING (Steps-Pre Line)
            ax.plot(months, sim_mean,
                    color=p_color,
                    linestyle='-',
                    drawstyle='steps-pre',
                    label=MODEL_LABELS[model_idx] + " Aralığı",  # Range Label
                    zorder=4)

            # 3. SCATTER POINTS and ERROR BARS

            # Scatter Points
            # ax.scatter(months[:-1], sim_mean[:-1],
            #            color=p_color,
            #            marker='o',
            #            alpha=0.8,
            #            s=20,
            #            zorder=5)

            # Error Bars
            # ax.errorbar(months[:-1], sim_mean[:-1],
            #             yerr=sim_std[:-1],
            #             fmt='none',
            #             capsize=3,
            #             linewidth=1.5,
            #             color=p_color,
            #             alpha=0.8,
            #             zorder=3)

        # --- Plot Constraints and Shading ---
        if const_lower_vals is not None:
            current_lower = const_lower_vals[node_idx]
            current_upper = const_upper_vals[node_idx]

            # Plot fixed constraint lines (Grey and Gold/Yellow)
            ax.axhline(current_upper, color="#777777", linestyle="--", linewidth=1.5, alpha=0.9, zorder=1, label='Üst Sınır')
            ax.axhline(current_lower, color="#777777", linestyle="--", linewidth=1.5, alpha=0.9, zorder=1, label='Alt Sınır')

            plot_lower_limit, plot_upper_limit = ax.get_ylim()

            lower_limit_val = min(current_lower * 0.95, plot_lower_limit)
            upper_limit_val = max(current_upper * 1.05, plot_upper_limit)

            # Shade the UNCONSTRAINED areas (Light Gray)
            ax.axhspan(current_upper, upper_limit_val, facecolor='#EAEAEA', alpha=0.9, zorder=0)
            ax.axhspan(lower_limit_val, current_lower, facecolor='#EAEAEA', alpha=0.9, zorder=0)

            ax.set_ylim(lower_limit_val, upper_limit_val)

        # --- Final Plot Styling ---

        ax.set_title(f'{node_name}', fontsize=14, pad=15)

        ax.set_xlabel("Aylar", fontsize=12)

        # --- TURKISH MONTH LABELS ---

        # The labels should be TR_MONTHS repeated for the 13th point
        x_labels = np.concatenate((TR_MONTHS, [TR_MONTHS[0]]))

        ax.set_xticks(np.arange(0, 13))
        ax.set_xticklabels(x_labels, rotation=0, ha='center')

        # --- END TURKISH MONTH LABELS ---
        # Set the custom limits based on your provided values:
        ax.set_xlim(-0.05, 12.05)
        ax.set_ylabel("Rezervuar Hacmi (hm$^3$)", fontsize=12)
        ax.grid(True, color='#DDDDDD', linestyle='--', axis='y', zorder=0)
        ax.grid(True, color='#DDDDDD', linestyle=':', axis='x', zorder=0)

        # Legend Creation
        model_handles = []
        for i in range(len(MODEL_LABELS)):
            # Line handle (represents the Range/Step)
            line_handle = plt.Line2D([0], [0], color=plot_colors[i]["line"], linewidth=2, linestyle='-', label=MODEL_LABELS[i] + " Aralığı")
            # Scatter handle (represents the Average dot)
            scatter_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_colors[i]["line"], markersize=8,
                                        label=MODEL_LABELS[i] + " Ortalaması")
            model_handles.extend([line_handle, scatter_handle])

        constraint_handles = [
            plt.Line2D([0], [0], color="#777777", linewidth=1.5, linestyle='-', label='Üst Sınır'),
            plt.Line2D([0], [0], color="#FFCC00", linewidth=1.5, linestyle='-', label='Alt Sınır')
        ]

        # ax.legend(handles=model_handles + constraint_handles,
        #           title="Hidrolojik Koşullar", loc="upper right", frameon=True, fontsize=8, ncol=2)

        # BİNLİK AYRACI UYGULAMASI
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_fixed))  # <-- Düzeltilmiş Fonksiyonu Kullanın
        # plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95, hspace=0.3, wspace=0.3)

        plt.tight_layout()
        plt.show()
        plt.close(fig)



# ---------------------------------------------------------------------------------
def thousands_formatter_fixed(x, pos):
    """1000 -> 1.000 veya 1000000 -> 1.000.000 formatını sağlar (Yerel ayardan bağımsız)."""
    s = f'{int(round(x)):d}'
    result = []
    for i, digit in enumerate(reversed(s)):
        if i > 0 and i % 3 == 0:
            result.append('.')
        result.append(digit)
    return "".join(reversed(result))
# ---------------------------------------------------------------------------------

def gemplot_scatter_collect(sim_outputs, obs_data, node_names, month_names, lower_limit=None, upper_limit=None, initial=None):
    """
    Plots a collective figure using the original boxplot structure but applies the
    steps-pre style to the mean line and adds the Turkish thousands separator
    to the Y-axis.
    """
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)
    num_plots = len(node_names)

    # Figür ayarı
    fig, axs = plt.subplots(1, num_plots, figsize=(12, 5)) # Adjusted size for better layout
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Normal Yıl (Mor)
    COLOR_NORMAL_LINE = "#993399"
    COLOR_NORMAL_SCATTER = "#6A1AAB"

    # Islak Yıl (Camgöbeği/Turkuaz)
    COLOR_WET_LINE = "#009999"
    COLOR_WET_SCATTER = "#007F7F"

    # Kurak Yıl (Turuncu/Kiremit)
    COLOR_DRY_LINE = "#FF7F00"
    COLOR_DRY_SCATTER = "#CC6600"

    # Ana Renk Listeleri güncelleniyor:
    model_colors = [COLOR_NORMAL_LINE, COLOR_WET_LINE, COLOR_DRY_LINE]
    model_dark_colors = [COLOR_NORMAL_SCATTER, COLOR_WET_SCATTER, COLOR_DRY_SCATTER]
    model_labels = ["Normal Yıl", "Islak Yıl", "Kuru Yıl"]

    for ax_idx, (ax, node_name) in enumerate(zip(axs, node_names)):
        node_idx = np.where(node_names == node_name)[0][0]

        # **Çizim Sırası: Islak, Kuru, Normal**
        for model_idx in [1, 2, 0]:
            current_color = model_colors[model_idx]
            current_dark_color = model_dark_colors[model_idx]

            sim_output = sim_outputs[model_idx]
            node_data = sim_output[:, node_idx, :]

            # --- VERİ HAZIRLIĞI ---
            df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
            df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()
            first_month_data = df[df['Month'] == '1'].copy()
            first_month_data['Month'] = '13'
            df = pd.concat([df, first_month_data], ignore_index=True)

            # Boxplot Stilleri
            boxprops = dict(linestyle='-', linewidth=1.2, alpha=0.2, facecolor="none", edgecolor=current_color)
            capprops = dict(linewidth=1.2, alpha=0.2, color=current_color)
            whiskerprops = dict(linewidth=1.2, alpha=0.2, color=current_color)
            medianprops = dict(linewidth=1.2, linestyle='-', alpha=0.2, color=current_color)
            meanprops = dict(marker='o', markersize=9, markerfacecolor=current_color, alpha=0.2)

            # **Seaborn Boxplot**
            sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                        width=0.7, boxprops=boxprops, whiskerprops=whiskerprops,
                        capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                        meanprops=meanprops)

            # **Ortalama Hesaplama ve Uzatma**
            sim = np.mean(node_data, axis=0)
            sim = np.concatenate((sim, [sim[0]]))
            original_months = np.arange(0, 13)

            # **Scatter Plot**
            # ax.scatter(original_months, sim, color=current_dark_color, marker='o', alpha=0.8, s=10, zorder=5)

            # 🔥 **ADIM ÇİZGİSİ (STEP LINE) ÇİZİMİ** 🔥
            # Kübik interpolasyon yerine steps-pre kullanılır.
            ax.plot(original_months, sim, color=current_color, linestyle='-', alpha=0.7, linewidth=1.5, zorder=4, drawstyle='steps-pre')

            # Y ekseni etiketi sadece ilk grafiğe
            ax.set(ylabel=None)
            ax.set_xlim(-0.05, 12.05)
            if ax_idx == 0:
                ax.set_ylabel("Rezervuar Hacmi (hm$^3$)", fontsize=11)


        # --- KISITLAR VE GÖLGELENDİRME ---
        if lower_limit is not None and upper_limit is not None and initial is not None:
            # Kısıt değerlerinin tek değer olduğu varsayılır.
            current_lower = lower_limit[node_idx][0] if lower_limit[node_idx].ndim > 0 else lower_limit[node_idx]
            current_upper = upper_limit[node_idx][0] if upper_limit[node_idx].ndim > 0 else upper_limit[node_idx]
            current_initial = initial[node_idx][0] if initial[node_idx].ndim > 0 else initial[node_idx]

            # Kısıt Çizgileri
            ax.axhline(current_upper, color="grey", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
            ax.axhline(current_lower, color="grey", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
            ax.axhline(current_initial, color="khaki", linestyle="--", linewidth=1, alpha=0.8, zorder=1)

            # Gölgelendirme
            plot_lower_limit, plot_upper_limit = ax.get_ylim()
            lower_limit_val = min(current_lower, plot_lower_limit)
            upper_limit_val = max(current_upper, plot_upper_limit)

            ax.axhspan(lower_limit_val, current_lower, facecolor='lightgrey', alpha=0.3, zorder=0)
            ax.axhspan(current_upper, upper_limit_val, facecolor='lightgrey', alpha=0.3, zorder=0)
            ax.set_ylim(lower_limit_val, upper_limit_val)

        # --- FİNAL STİL VE EKSENLER ---
        ax.set_title(f'{node_name}', fontsize=12, pad=15)
        ax.set_xlabel("Aylar", fontsize=10)

        # X EKSENİ (Orijinal month_names girdisi kullanılır)
        x_labels = np.concatenate((month_names, [month_names[0]]))
        ax.set_xticks(original_months)
        ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)

        # 🔥 Y EKSENİ (BİNLİK AYRAÇ UYGULAMASI) 🔥
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_fixed))

        ax.grid(True, color='lightgrey', linestyle='--')

    # **Ortak Lejant (Alt Boşluğa)**
    # Lejant Handle'ları (Türkçe etiketler ve renkler)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=model_labels[i],
                          markerfacecolor=model_colors[i], markersize=10) for i in range(len(model_labels))]

    # Kısıt handle'ları
    constraint_handles = [
        plt.Line2D([0], [0], color="grey", linewidth=1.2, linestyle='-', label="Kısıt (Üst/Alt)"),
        plt.Line2D([0], [0], color="khaki", linewidth=0.0, linestyle='-', label="Başlangıç Hacmi")
    ]

    # fig.legend(handles=handles + constraint_handles, title="Hidrolojik Koşullar", loc="lower center",
    #            bbox_to_anchor=(0.5, -0.05), # İstenen konum
    #            ncol=len(model_labels) + 2, frameon=False, fontsize=10)

    plt.subplots_adjust(bottom=0.25, top=0.95, wspace=0.25)
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    plt.show()


def gemplot_scatter_single(sim_outputs, obs_data, node_names, month_names, lower_limit=None, upper_limit=None, initial=None):
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)
    num_plots = len(node_names)
    fig, axs = plt.subplots(1, num_plots, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    model_color = "#993399"  # Purple for Normal Year
    model_label = "Normal Year"

    for ax_idx, (ax, node_name) in enumerate(zip(axs, node_names)):
        node_idx = np.where(node_names == node_name)[0][0]

        # Extract the data for the Normal Year scenario (assuming it's at index 0)
        sim_output = sim_outputs[0]  # Only one scenario
        node_data = sim_output[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Duplicate first month's data at the end for smooth 13-month cycle
        first_month_data = df[df['Month'] == '1'].copy()
        first_month_data['Month'] = '13'
        df = pd.concat([df, first_month_data], ignore_index=True)

        # **Seaborn Boxplot**
        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.7, boxprops=dict(facecolor="none", edgecolor=model_color, linewidth=1.2, alpha=0.4),
                    whiskerprops=dict(color=model_color, linewidth=1.2, alpha=0.4),
                    capprops=dict(color=model_color, linewidth=1.2, alpha=0.4),
                    flierprops=flierprops,
                    medianprops=dict(color=model_color, linestyle='-', linewidth=1.2, alpha=0.4))

        # **Plot Simulated Mean**
        sim = np.mean(node_data, axis=0)  # Mean for 12 months
        sim = np.concatenate((sim, [sim[0]]))  # Extend to 13 months for cyclic plotting

        # **Scatter Plot**
        ax.scatter(np.arange(0, 13), sim, color=model_color, marker='o', alpha=0.8, s=10, zorder=5)

        # **Cubic Interpolation for Smooth Line**
        original_months = np.arange(0, 13)  # Matches sim indices
        smoothed_months = np.linspace(0, 12, 100)  # Smooths between months
        interpolation_fn = interp1d(original_months, sim, kind='cubic', fill_value="extrapolate")
        smoothed_data = interpolation_fn(smoothed_months)
        smoothed_data = np.clip(smoothed_data, 0, upper_limit[node_idx])

        ax.plot(smoothed_months, smoothed_data, color=model_color, linestyle='-', alpha=0.7, linewidth=1.5, zorder=4)

        # **Plot Constraints**
        if lower_limit is not None and upper_limit is not None and initial is not None:
            lower_limit = np.concatenate((lower_limit, [lower_limit[0]]))
            upper_limit = np.concatenate((upper_limit, [upper_limit[0]]))
            initial = np.concatenate((initial, [initial[0]]))

            ax.axhline(upper_limit[node_idx], color="grey", linestyle="-", linewidth=1.2, alpha=0.8, zorder=1)
            ax.axhline(lower_limit[node_idx], color="grey", linestyle="-", linewidth=1.2, alpha=0.8, zorder=1)
            ax.axhline(initial[node_idx], color="khaki", linestyle="-", linewidth=1, alpha=0.8, zorder=1)

        ax.set_title(f'{node_name}')
        ax.set_xlabel("Months")
        ax.set_xticklabels(np.concatenate((month_names, [month_names[0]])))
        ax.grid(True, color='lightgrey', linestyle='--')

    # **Legend**
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=model_label,
                          markerfacecolor=model_color, markersize=10)]
    fig.legend(handles=handles, title="Hydrological Condition", loc="lower center", bbox_to_anchor=(0.5, -0.1),
               ncol=1, frameon=False)

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()

def plot_yearly_total_energy_with_observation(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH):
    """
    Plots the yearly total energy (sum of all nodes and months) for each scenario,
    including observation data as a reference.

    Parameters:
    - scenarios: List of tuples [(generated_energy_gwh, label, color), ...] for each scenario.
    - OBS_EN_GWH: Observed energy data (2D array with dimensions nodes x months).
    - OBS_EN_FIRM_GWH: Observed firm energy data (2D array with dimensions nodes x months).
    """
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors = ["#D7BDE2", "#F9E79F", "#76D7C4", "#AED6F1"]
    observation_color = "#48565E"
    observation_lighter_color = "#D3D3D3"

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar width for side-by-side alignment
    bar_width = 0.5  # Make bars wider (default was 0.2)
    x = np.linspace(0, scenario_count, scenario_count + 1)  # Base x positions for bars
    #
    # Add observation data
    total_obs_firm_energy = OBS_EN_FIRM_GWH.sum()
    total_obs_secondary_energy = (OBS_EN_GWH - OBS_EN_FIRM_GWH).clip(min=0).sum()

    ax.bar(
        x[0], total_obs_firm_energy, width=bar_width, color=observation_color,
        label="Observed Firm Energy"
    )
    ax.bar(
        x[0], total_obs_secondary_energy, width=bar_width, color=observation_lighter_color,
        bottom=total_obs_firm_energy, label="Observed Secondary Energy"
    )

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes and months
        total_firm_energy = EN_FIRM_GWH.sum()
        total_secondary_energy = EN_SECN_GWH.sum()

        # Plot stacked bars for the current scenario
        ax.bar(
            x[i + 1], total_firm_energy, width=bar_width, color=colors[i],
            label=f"{label} Firm Energy" if i == 0 else None
        )
        ax.bar(
            x[i + 1], total_secondary_energy, width=bar_width, color=lighter_colors[i],
            bottom=total_firm_energy, label=f"{label} Secondary Energy" if i == 0 else None
        )

    # Configure plot
    ax.set_title("Yearly Total Energy Generated Across All Nodes (Including Observations)")
    ax.set_ylabel("Total Energy (GWh)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Observed"] + [label for _, label, _ in scenarios])
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    # # Add legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_yearly_total_energy_without_observation(scenarios):
    """
    Plots the yearly total energy (sum of all nodes and months) for each scenario.

    Parameters:
    - scenarios: List of tuples [(generated_energy_gwh, label, color), ...] for each scenario.
    """
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors = ["#D7BDE2", "#F9E79F", "#76D7C4", "#AED6F1"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar width for side-by-side alignment
    bar_width = 0.5  # Adjust bar width as needed
    x = np.arange(scenario_count)  # Base x positions for bars

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes and months
        total_firm_energy = EN_FIRM_GWH.sum()
        total_secondary_energy = EN_SECN_GWH.sum()

        # Plot stacked bars for the current scenario
        ax.bar(
            x[i], total_firm_energy, width=bar_width, color=colors[i],
            label=f"{label} Firm Energy" if i == 0 else None
        )
        ax.bar(
            x[i], total_secondary_energy, width=bar_width, color=lighter_colors[i],
            bottom=total_firm_energy, label=f"{label} Secondary Energy" if i == 0 else None
        )

    # Configure plot
    # ax.set_title("Yearly Total Energy Generated Across All Nodes")
    ax.set_ylabel("Total Energy (GWh)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in scenarios])
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_yearly_total_energy_without_observation_turkish(scenarios):
    """
    Plots the yearly total energy (sum of all nodes and months) for each scenario.

    Parameters:
    - scenarios: List of tuples [(generated_energy_gwh, label, color), ...] for each scenario.
    """
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors = ["#D7BDE2", "#F9E79F", "#76D7C4", "#AED6F1"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar width for side-by-side alignment
    bar_width = 0.5  # Adjust bar width as needed
    x = np.arange(scenario_count)  # Base x positions for bars

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes and months
        total_firm_energy = EN_FIRM_GWH.sum()
        total_secondary_energy = EN_SECN_GWH.sum()

        # Plot stacked bars for the current scenario
        ax.bar(
            x[i], total_firm_energy, width=bar_width, color=colors[i],
            label=f"{label} Firm Energy" if i == 0 else None
        )
        ax.bar(
            x[i], total_secondary_energy, width=bar_width, color=lighter_colors[i],
            bottom=total_firm_energy, label=f"{label} Secondary Energy" if i == 0 else None
        )

    # Configure plot
    # ax.set_title("Yearly Total Energy Generated Across All Nodes")
        ax.set_xticks(x)
    ax.set_ylabel("Enerji Üretimi (GWh)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticklabels([label for _, label, _ in scenarios])
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_fixed))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def gemplot_generated_energy_obs_bar(GEN_EN_GWH, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    node_count = 5
    nodes = [4, 5, 7, 8, 9]

    EN_SECN_GWH = np.maximum(GEN_EN_GWH - OBS_EN_FIRM_GWH, 0)
    EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, GEN_EN_GWH)

    # Calculate OBS_EN_SECN_GWH
    OBS_EN_SECN_GWH = np.maximum(OBS_EN_GWH - OBS_EN_FIRM_GWH, 0)

    colors = ["#F9E79F", "orange"]  # Yellow and orange
    observation_color = "#48565E"
    observation_lighter_color = "#D3D3D3"

    fig, axs = plt.subplots(1, node_count, figsize=(16, 4), sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, node in enumerate(nodes):
        idx = node - 1
        ax = axs[i]

        # Calculate bar width and positions
        bar_width = 0.4
        sim_positions = np.arange(len(months)) - bar_width / 2
        obs_positions = np.arange(len(months)) + bar_width / 2

        # Plot simulated firm energy
        ax.bar(sim_positions, EN_FIRM_GWH[idx, :], bar_width, color=colors[1], label="Simulated Firm Energy")
        # Plot simulated secondary energy
        ax.bar(sim_positions, EN_SECN_GWH[idx, :], bar_width, bottom=EN_FIRM_GWH[idx, :], color=colors[0], label="Simulated Secondary Energy")
        # Plot observed firm energy
        ax.bar(obs_positions, OBS_EN_FIRM_GWH[idx, :], bar_width, color=observation_color, label="Observed Firm Energy")
        # Plot observed secondary energy
        ax.bar(obs_positions, OBS_EN_SECN_GWH[idx, :], bar_width, bottom=OBS_EN_FIRM_GWH[idx, :], color=observation_lighter_color, label="Observed Secondary Energy")

        # Add labels and title (only for the first subplot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper left")

        ax.set_title(titles[idx])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey')
        ax.tick_params(axis='both', which='major', labelsize=9)
        month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        ax.set_xticks(np.arange(len(months)))  # Set x-ticks at integer positions
        ax.set_xticklabels(month_labels)

    axs[0].set_ylabel("Generated Energy (GWh)")

    plt.tight_layout()
    plt.show()

def gemplot_generated_energy_obs_bar_noseperation(GEN_EN_GWH, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    node_count = 5
    nodes = [4, 5, 7, 8, 9]

    EN_SECN_GWH = np.maximum(GEN_EN_GWH - OBS_EN_FIRM_GWH, 0)
    EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, GEN_EN_GWH)

    # Calculate OBS_EN_SECN_GWH
    OBS_EN_SECN_GWH = np.maximum(OBS_EN_GWH - OBS_EN_FIRM_GWH, 0)

    # Calculate total energy for simulated and observed data
    sim_total_energy = EN_FIRM_GWH + EN_SECN_GWH
    obs_total_energy = OBS_EN_FIRM_GWH + OBS_EN_SECN_GWH

    colors = ["#F9E79F", "orange"]  # Yellow and orange
    observation_color = "#48565E"
    observation_lighter_color = "#D3D3D3"

    fig, axs = plt.subplots(1, node_count, figsize=(16, 4), sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, node in enumerate(nodes):
        idx = node - 1
        ax = axs[i]

        # Calculate bar width and positions
        bar_width = 0.4
        sim_positions = np.arange(len(months)) - bar_width / 2
        obs_positions = np.arange(len(months)) + bar_width / 2

        # Plot simulated firm energy
        ax.bar(sim_positions, sim_total_energy[idx, :], bar_width, color=colors[1], label="Simulated Total Energy")
        # Plot observed total energy
        ax.bar(obs_positions, obs_total_energy[idx, :], bar_width, color=observation_color, label="Observed Total Energy")

        # Add labels and title (only for the first subplot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper left")

        ax.set_title(titles[idx])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey')
        ax.tick_params(axis='both', which='major', labelsize=9)
        month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        ax.set_xticks(np.arange(len(months)))  # Set x-ticks at integer positions
        ax.set_xticklabels(month_labels)

    axs[0].set_ylabel("Generated Energy (GWh)")

    plt.tight_layout()
    plt.show()


def gemplot_generated_energy_obs_line(GEN_EN_GWH, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    node_count = 5
    nodes = [4, 5, 7, 8, 9]

    EN_SECN_GWH = np.maximum(GEN_EN_GWH - OBS_EN_FIRM_GWH, 0)
    EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, GEN_EN_GWH)

    # Calculate OBS_EN_SECN_GWH
    OBS_EN_SECN_GWH = np.maximum(OBS_EN_GWH - OBS_EN_FIRM_GWH, 0)

    # Calculate total energy for simulated and observed data
    sim_total_energy = EN_FIRM_GWH + EN_SECN_GWH
    obs_total_energy = OBS_EN_FIRM_GWH + OBS_EN_SECN_GWH

    colors = ["PURPLE", "#C48630","#F1C36B", "#EFB743"]  # Yellow and orange

    fig, axs = plt.subplots(1, node_count, figsize=(16, 4), sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, node in enumerate(nodes):
        idx = node - 1
        ax = axs[i]

        # Plot simulated total energy as an area plot
        ax.fill_between(months, 0, sim_total_energy[idx, :], color=colors[2], alpha=0.9)

        # Plot observed total energy as an area plot
        ax.fill_between(months, 0, obs_total_energy[idx, :], color=colors[3], alpha=0.9)


        # Add line plots with 'o' markers for each series
        ax.plot(months, obs_total_energy[idx, :], color=colors[1], marker='o', linestyle='--',
                linewidth=1, markersize=4, alpha=0.9, label="Reference")
        ax.plot(months, sim_total_energy[idx, :], color=colors[0], marker='o', linestyle='-',
                linewidth=1,  markersize=4, alpha=0.9, label="Simulation")

        # Add labels and title (only for the first subplot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper left")

        ax.set_title(titles[idx])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major')
        month_labels = ["Oc", "No", "De", "Ja", "Fe", "Ma", "Ap", "Ma", "Jn", "Jl", "Au", "Se"]
        # month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        ax.set_xticks(np.arange(len(months)))  # Set x-ticks at integer positions
        ax.set_xticklabels(month_labels)
        # Set y-axis limits to start from 0
        max_value = np.max(np.maximum(obs_total_energy, sim_total_energy))  # Determine the maximum value in the data
        ax.set_ylim(0, 1.1*max_value)

    axs[0].set_ylabel("Generated Energy (GWh)")

    plt.tight_layout()
    plt.show()

def gemplot_generated_energy_scenarios(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    """Plots generated energy for different scenarios along with observed values."""

    # Constants
    node_count = 5
    nodes = [4, 5, 7, 8, 9]  # Indices of nodes (1-based)
    colors = ["purple", "orange", "#008080", "#00BFFF"]  # Base colors for FIRM_GWH

    lighter_colors = ["#D7BDE2", "#F9E79F", "#76D7C4", "#AED6F1"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Remove the last item from titles to match data dimensions
    titles = titles[:-1]

    # Create the figure and axes
    fig, axs = plt.subplots(1, node_count, figsize=(24, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    # Iterate through each node
    for j, node in enumerate(nodes):
        ax = axs[j]

        # Prepare bar width for side-by-side alignment
        bar_width = 0.2  # Increased width for larger bars
        x = np.arange(len(months))  # Base x positions for bars

        # Iterate through scenarios
        for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
            # Reshape data to 13x12 matrices
            EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
            EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

            # Extract data for the current node (adjusted for 0-based indexing)
            node_index = node - 1
            EN_FIRM_GWH_node = EN_FIRM_GWH[node_index, :]
            EN_SECN_GWH_node = EN_SECN_GWH[node_index, :]

            # Create a DataFrame for the current scenario and node
            data = {
                "Months": months,
                "Firm Energy (GWh)": EN_FIRM_GWH_node,
                "Secondary Energy (GWh)": EN_SECN_GWH_node
            }
            df = pd.DataFrame(data)

            # Adjust x positions for side-by-side bars
            bar_positions = x + i * bar_width

            # Plot stacked bars for the current scenario
            ax.bar(
                bar_positions, EN_FIRM_GWH_node,
                width=bar_width, color=colors[i],
                label=f"{label} - Firm"
            )
            ax.bar(
                bar_positions, EN_SECN_GWH_node,
                width=bar_width, color=lighter_colors[i],
                bottom=EN_FIRM_GWH_node,
                label=f"{label} - Secondary"
            )

            # # Add observed values as a scatter plot
            # OBS_EN_GWH_node = OBS_EN_GWH[node_index, :]
            # ax.scatter(
            #     x + (scenario_count - 1) * bar_width / 2, OBS_EN_GWH_node,
            #     edgecolor="grey", facecolor="grey",
            #     s=20, zorder=4, alpha=0.8
            # )

            # Configure the subplot
            ax.set_title(titles[node_index])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey')
        ax.tick_params(axis='both', which='major', labelsize=11)

        # Set month labels
        month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        ax.set_xticks(x + (scenario_count - 1) * bar_width / 2)
        ax.set_xticklabels(month_labels)

    axs[0].set_ylabel("Generated Energy (GWh)")

    # Adding the legend only once for each scenario
    # Add labels only once by using a set for unique legend labels
    handles, labels = [], []
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Add legend entry for firm and secondary energy of each scenario
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))
        labels.append(f"{label} - Firm")
        labels.append(f"{label} - Secondary")

    # # Add observed energy legend entry
    # handles.append(plt.Line2D([0], [0], color="grey", lw=4))
    # labels.append("Observed Energy")

    # Add the custom legend at the bottom center
    fig.legend(handles, labels, loc="lower center", ncol=4)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def gemplot_generated_energy_scenarios_line(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    """Plots generated energy for different scenarios along with observed values using line plots."""
    # Constants
    node_count = 5
    nodes = [4, 5, 7, 8, 9]  # Indices of nodes (1-based)
    colors = ["purple", "orange", "#008080", "#00BFFF"]  # Colors for scenarios
    markers = ['o', 's', 'D', '^']  # Different markers for each scenario

    if len(scenarios) != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Remove the last item from titles to match data dimensions
    titles = [title + " HEPP" for title in titles[:-1]]

    # Create the figure and axes
    fig, axs = plt.subplots(1, node_count, figsize=(24, 6), sharey=True, dpi=200)
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    # Iterate through each node
    for j, node in enumerate(nodes):
        ax = axs[j]
        node_index = node - 1

        # Iterate through scenarios
        for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
            # Reshape data to 13x12 matrices
            total_energy = generated_energy_gwh.reshape(13, 12)

            # Extract data for the current node (adjusted for 0-based indexing)
            total_energy_node = total_energy[node_index, :]

            # Interpolate data for smooth plotting
            interpolation_fn = interp1d(np.arange(1, 13), total_energy_node, kind='cubic')
            smoothed_months = np.linspace(1, 12, num=100)
            sim_mean_smooth = interpolation_fn(smoothed_months)
            ax.plot(smoothed_months, sim_mean_smooth, linewidth=1.5, linestyle='-', color=colors[i], alpha=0.6,  label=label)

            # Plot scatter points for each month
            ax.scatter(np.arange(1, 13), total_energy_node,  marker=markers[i], s=40, alpha=0.8, color=colors[i])

        # Add observed energy as a separate line
        # OBS_EN_GWH_node = OBS_EN_GWH[node_index, :]
        # ax.plot(months, OBS_EN_GWH_node, marker='s', linestyle='--', color='black', label='Observed Energy')

        # Configure the subplot
        ax.set_title(titles[node_index])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_xticks(np.arange(1, 13))
        month_labels = ["Oc", "Nv", "Dc", "Jn", "Fb", "Mr", "Ap", "My", "Ju", "Jl", "Ag", "Sp"]
        ax.set_xticklabels(month_labels)
        ax.set_title(titles[node_index], fontsize=14)  # Increase font size for titles

    axs[0].set_ylabel("Generated Energy (GWh)")

    # Add legend at the bottom center
    # fig.legend(loc="lower center", ncol=5)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def gemplot_generated_energy_scenarios_line_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    """Plots generated energy for different scenarios along with observed values using line plots."""
    # Constants
    node_count = 5
    nodes = [4, 5, 7, 8, 9]  # Indices of nodes (1-based)
    colors = ["purple", "orange", "#008080", "#00BFFF"]  # Colors for scenarios
    markers = ['o', 's', 'D', '^']  # Different markers for each scenario

    if len(scenarios) != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Remove the last item from titles to match data dimensions
    titles = [title + " HES" for title in titles[:-1]]

    # Create the figure and axes
    fig, axs = plt.subplots(1, node_count, figsize=(24, 6), sharey=True, dpi=200)
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    # Iterate through each node
    for j, node in enumerate(nodes):
        ax = axs[j]
        node_index = node - 1

        # Iterate through scenarios
        for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
            # Reshape data to 13x12 matrices
            total_energy = generated_energy_gwh.reshape(13, 12)

            # Extract data for the current node (adjusted for 0-based indexing)
            total_energy_node = total_energy[node_index, :]

            # Interpolate data for smooth plotting
            interpolation_fn = interp1d(np.arange(1, 13), total_energy_node, kind='cubic')
            smoothed_months = np.linspace(1, 12, num=100)
            sim_mean_smooth = interpolation_fn(smoothed_months)
            ax.plot(np.arange(1, 13), total_energy_node, linewidth=1.5, linestyle='-', color=colors[i], alpha=0.6,  label=label)

            # Plot scatter points for each month
            ax.scatter(np.arange(1, 13), total_energy_node,  marker=markers[i], s=40, alpha=0.8, color=colors[i])

        # Add observed energy as a separate line
        # OBS_EN_GWH_node = OBS_EN_GWH[node_index, :]
        # ax.plot(months, OBS_EN_GWH_node, marker='s', linestyle='--', color='black', label='Observed Energy')

        # Configure the subplot
        ax.set_title(titles[node_index])
        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='y', which='major', labelsize=14)
        ax.set_xticks(np.arange(1, 13))
        month_labels = ["Ek", "Ks", "Ar", "Oc", "Şb", "Mr", "Ns", "My", "Hz", "Tm", "Ağ", "Ey"]
        ax.set_xticklabels(month_labels)
        ax.set_title(titles[node_index], fontsize=16)  # Increase font size for titles

        # 🟢 Başlık ve grafik arasına boşluk ekle (pad parametresi ile)
        ax.set_title(titles[node_index], fontsize=16, pad=15)  # <--- pad eklendi (20 px boşluk)

    axs[0].set_ylabel("Enerji Üretimi (GWh)", fontsize=16)

    # Add legend at the bottom center
    # fig.legend(loc="lower center", ncol=5)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def gemplot_generated_energy_scenarios_single_plots_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, titles, months):
    """
    Plots generated energy for different scenarios along with observed values
    using separate line plots for each node (HES).
    Months are written in full Turkish, and there's more space between title and plot.
    """
    # Constants
    node_count = 5
    nodes = [4, 5, 7, 8, 9]  # Indices of nodes (1-based)
    colors = ["purple", "orange", "#008080", "#00BFFF"]  # Colors for scenarios
    markers = ['o', 's', 'D', '^']  # Different markers for each scenario

    # Tam Türkçe ay isimleri
    full_month_labels = [
        "Ekim", "Kasım", "Aralık", "Ocak", "Şubat", "Mart",
        "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül"
    ]

    if len(scenarios) != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Remove the last item from titles and append ' HES' for consistency
    plot_titles = [title + " HES" for title in titles[:-1]]

    # Matplotlib'de ondalık (virgül) ve binlik (nokta) ayracı tanımlayan özel bir formatlayıcı
    class TurkishFormatter(ticker.Formatter):
        def __call__(self, x, pos=None):
            # Sayıyı binlik nokta ve ondalık virgül ile biçimlendirir
            # Örnek: 1234.56 -> 1.234,56

            # Öncelikle sayıyı yuvarla (grafiğe göre ondalık basamak sayısı ayarlanabilir)
            rounded_x = round(x, 2)

            # Tam kısmı ve ondalık kısmı ayır
            integer_part = int(rounded_x)
            decimal_part = f"{rounded_x:.2f}".split('.')[-1]  # 2 ondalık basamak için

            # Tam kısmı binlik nokta ile formatla
            formatted_integer = f"{integer_part:,}".replace(",", ".")

            # Eğer ondalık kısım sıfırlardan oluşuyorsa (örn: 123.00), ondalık kısmı at
            if decimal_part == '0' or not decimal_part:
                return formatted_integer
            else:
                # Tüm parçaları birleştir (nokta binlik, virgül ondalık)
                return f"{formatted_integer},{decimal_part}"

    # Formatlayıcı örneğini oluştur
    turkish_formatter = TurkishFormatter()
    # --- Her Bir Düğüm İçin Ayrı Bir Figür Oluşturma ---

    # Iterate through each node
    for j, node in enumerate(nodes):
        node_index = node - 1
        current_title = plot_titles[node_index]

        # Her bir düğüm için yeni bir figür ve eksen oluştur
        fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=200) # figsize biraz büyütüldü

        # Iterate through scenarios
        for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
            # Reshape data to 13x12 matrices (Assuming the first dimension is node-related)
            total_energy = generated_energy_gwh.reshape(13, 12)

            # Extract data for the current node (adjusted for 0-based indexing)
            total_energy_node = total_energy[node_index, :]

            # Interpolate data for smooth plotting
            interpolation_fn = interp1d(np.arange(1, 13), total_energy_node, kind='cubic')
            smoothed_months = np.linspace(1, 12, num=100)
            sim_mean_smooth = interpolation_fn(smoothed_months)

            # Plot the smoothed line
            ax.plot(np.arange(1, 13), total_energy_node, linewidth=1.5, linestyle='-',
                    color=colors[i], alpha=0.7, label=label)

            # Plot scatter points for each month
            ax.scatter(np.arange(1, 13), total_energy_node, marker=markers[i],
                       s=50, alpha=0.9, color=colors[i], zorder=3)

        # --- Obsverved Data Plot (Opsiyonel: Eğer OBS_EN_GWH kullanılacaksa) ---
        # OBS_EN_GWH_node = OBS_EN_GWH[node_index, :]
        # ax.plot(np.arange(1, 13), OBS_EN_GWH_node, marker='s', linestyle='--',
        #         color='black', linewidth=2, label='Observed Energy', zorder=2)

        # Y Eksenine Türkçe Sayı Formatlayıcısını Uygula
        ax.yaxis.set_major_formatter(turkish_formatter)
        # --- Configure the Single Plot ---
        # Başlık ve grafik arasına boşluk eklemek için pad parametresi kullanıldı
        ax.set_title(current_title, fontsize=18, pad=15) # Başlık fontu büyütüldü, pad ile boşluk eklendi
        ax.set_ylabel("Enerji Üretimi (GWh)", fontsize=16)
        ax.set_xlabel("Aylar", fontsize=16)

        ax.set_axisbelow(True)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.8)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(full_month_labels) # Ayların tam isimleri ve eğimli yazımrotation=45, ha='right'

        # Add legend for the current plot
        # ax.legend(loc='best', fontsize=12)

        # Adjust layout to prevent labels from overlapping and show the current plot
        plt.tight_layout()
        plt.show() # Her figürü ayrı ayrı göster



def plot_total_energy(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, months):
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors =  ["#C79ED4", "#FFD380", "#66B2B2", "#66D9FF"]
    # ["#9b7f9e", "#D68E4E", "#6F8E8B", "#7BB7D6"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Bar width for side-by-side alignment
    bar_width = 0.2
    x = np.arange(len(months))  # Base x positions for bars

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Initialize lists for the legend
    handles = []
    labels = []

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices (assuming it represents something other than months)
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes (sum over axis 0)
        total_firm_energy = EN_FIRM_GWH.sum(axis=0)
        total_secondary_energy = EN_SECN_GWH.sum(axis=0)

        # Adjust x positions for side-by-side bars
        bar_positions = x + i * bar_width

        # Plot stacked bars for the current scenario
        ax.bar(
            bar_positions, total_firm_energy, width=bar_width, color=colors[i], alpha=1,
            label=f"{label} Firm Energy"
        )
        ax.bar(
            bar_positions, total_secondary_energy, width=bar_width, color=lighter_colors[i], alpha=1,
            bottom=total_firm_energy, label=f"{label} Secondary Energy"
        )

        # Append handles and labels for the legend
        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))
        labels.append(f"{label} - Secondary")
        labels.append(f"{label} - Firm")

    # Configure plot
    # ax.set_title("Total Energy Generated Across All Nodes")
    ax.set_ylabel("Generated Energy (GWh)")

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    ax.set_xticklabels(month_labels)
    ax.set_xticks(x + (scenario_count - 1) * bar_width / 2)
    ax.set_xticklabels(month_labels)
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    # Initialize lists for the legend
    handles = []
    labels = []

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices (assuming it represents something other than months)
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes (sum over axis 0)
        total_firm_energy = EN_FIRM_GWH.sum(axis=0)
        total_secondary_energy = EN_SECN_GWH.sum(axis=0)

        # Adjust x positions for side-by-side bars
        bar_positions = x + i * bar_width

        # Plot stacked bars for the current scenario
        ax.bar(
            bar_positions, total_firm_energy, width=bar_width, color=colors[i], alpha=0.8,
            label=f"{label} Firm Energy"
        )
        ax.bar(
            bar_positions, total_secondary_energy, width=bar_width, color=lighter_colors[i], alpha=0.6,
            bottom=total_firm_energy, label=f"{label} Secondary Energy"
        )

        # Add primary (firm) color first, followed by secondary color to the legend
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))  # Primary color (Firm)
        labels.append(f"{label} - Firm")

        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))  # Secondary color
        labels.append(f"{label} - Secondary")

    # Add legend at the bottom in two rows
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)  # 2 rows, 4 columns

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_total_energy_turkish(scenarios, OBS_EN_GWH, OBS_EN_FIRM_GWH, months):
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors =  ["#C79ED4", "#FFD380", "#66B2B2", "#66D9FF"]
    # ["#9b7f9e", "#D68E4E", "#6F8E8B", "#7BB7D6"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Bar width for side-by-side alignment
    bar_width = 0.2
    x = np.arange(len(months))  # Base x positions for bars

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Initialize lists for the legend
    handles = []
    labels = []

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices (assuming it represents something other than months)
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes (sum over axis 0)
        total_firm_energy = EN_FIRM_GWH.sum(axis=0)
        total_secondary_energy = EN_SECN_GWH.sum(axis=0)

        # Adjust x positions for side-by-side bars
        bar_positions = x + i * bar_width

        # Plot stacked bars for the current scenario
        ax.bar(
            bar_positions, total_firm_energy, width=bar_width, color=colors[i], alpha=1,
            label=f"{label} Güvenilir (Firm) Enerji"
        )
        ax.bar(
            bar_positions, total_secondary_energy, width=bar_width, color=lighter_colors[i], alpha=1,
            bottom=total_firm_energy, label=f"{label} İkincil (Sekonder) Enerji"
        )

        # Append handles and labels for the legend
        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))
        labels.append(f"{label} - Sekonder")
        labels.append(f"{label} - Firm")

    # Configure plot
    # ax.set_title("Total Energy Generated Across All Nodes")
    ax.set_ylabel("Enerji Üretimi (GWh)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    month_labels = ["Eki", "Kas", "Ara", "Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl"]
    ax.set_xticklabels(month_labels)
    ax.set_xticks(x + (scenario_count - 1) * bar_width / 2)
    ax.set_xticklabels(month_labels)
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    # Initialize lists for the legend
    handles = []
    labels = []

    # Iterate through scenarios
    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices (assuming it represents something other than months)
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes (sum over axis 0)
        total_firm_energy = EN_FIRM_GWH.sum(axis=0)
        total_secondary_energy = EN_SECN_GWH.sum(axis=0)

        # Adjust x positions for side-by-side bars
        bar_positions = x + i * bar_width

        # Plot stacked bars for the current scenario
        ax.bar(
            bar_positions, total_firm_energy, width=bar_width, color=colors[i], alpha=0.8,
            label=f"{label} Firm Energy"
        )
        ax.bar(
            bar_positions, total_secondary_energy, width=bar_width, color=lighter_colors[i], alpha=0.6,
            bottom=total_firm_energy, label=f"{label} Secondary Energy"
        )

        # Add primary (firm) color first, followed by secondary color to the legend
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))  # Primary color (Firm)
        labels.append(f"{label} - Firm")

        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))  # Secondary color
        labels.append(f"{label} - Secondary")

    # Add legend at the bottom in two rows
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)  # 2 rows, 4 columns

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_yearly_total_energy(scenarios, OBS_EN_FIRM_GWH):
    """
    Plots the yearly total energy (sum of all nodes and months) for each scenario.

    Parameters:
    - scenarios: List of tuples [(generated_energy_gwh, label, color), ...] for each scenario.
    - OBS_EN_FIRM_GWH: Observed firm energy data (2D array with dimensions nodes x months).
    """
    # Colors for bars
    colors = ["purple", "orange", "#008080", "#00BFFF"]
    lighter_colors = ["#D7BDE2", "#F9E79F", "#76D7C4", "#AED6F1"]

    scenario_count = len(scenarios)
    if scenario_count != 4:
        raise ValueError("The 'scenarios' parameter must contain exactly 4 scenarios.")

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width for side-by-side alignment
    bar_width = 0.4
    x = np.arange(scenario_count)  # Base x positions for bars (1 bar per scenario)

    # Iterate through scenarios
    total_firm_energies = []
    total_secondary_energies = []

    for i, (generated_energy_gwh, label, color) in enumerate(scenarios):
        # Reshape data to 13x12 matrices
        EN_SECN_GWH = np.maximum(generated_energy_gwh - OBS_EN_FIRM_GWH, 0).reshape(13, 12)
        EN_FIRM_GWH = np.maximum(OBS_EN_FIRM_GWH, generated_energy_gwh).reshape(13, 12)

        # Calculate total energy for all nodes and months
        total_firm_energy = EN_FIRM_GWH.sum()
        total_secondary_energy = EN_SECN_GWH.sum()

        total_firm_energies.append(total_firm_energy)
        total_secondary_energies.append(total_secondary_energy)

        # Plot stacked bars for the current scenario
        ax.bar(
            x[i], total_firm_energy, width=bar_width, color=colors[i],
            label=f"{label} Firm Energy"
        )
        ax.bar(
            x[i], total_secondary_energy, width=bar_width, color=lighter_colors[i],
            bottom=total_firm_energy, label=f"{label} Secondary Energy"
        )

    # Configure plot
    ax.set_title("Yearly Total Energy Generated Across All Nodes")
    ax.set_ylabel("Total Energy (GWh)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in scenarios])
    ax.grid(True, axis='y', color='lightgrey', linestyle='--', linewidth=0.7)

    # Add legend
    handles, labels = [], []
    for i, (_, label, _) in enumerate(scenarios):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=lighter_colors[i]))
        labels.append(f"{label} Firm Energy")
        labels.append(f"{label} Secondary Energy")

    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def get_generated_energy_for_scenario(decision_variables, objective_function_values, weights):
    """Calculates the generated energy for a given scenario and weights."""

    # Calculate pseudo weights
    _, pseudo_weights = PseudoWeights(weights).do(objective_function_values, return_pseudo_weights=True)

    # Get the best solution index for the given weights
    best_solution_index = PseudoWeights(weights).do(objective_function_values)

    # Extract the decision variables for the best solution
    decision_variables = np.array([float(val) for val in decision_variables[best_solution_index]]).reshape(13, 12)

    # Get simulation results
    total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    result = simulation_model(decision_variables, K_1, K_2, K_3, K_SC)

    # Extract generated energy
    generated_energy_gwh = result[10]  # Assuming GEN_EN_GWH is at index 10
    return generated_energy_gwh


def gemplot_irrigation_box_BKP(sim_outputs, obs_data, node_names, month_names, demand_min, demand_max):
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)

    num_plots = len(node_names)

    # Create subplots with 2 rows and 4 columns
    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    # Color palette
    boxplot_color = "#669966"
    boxplot_line_color = "#004400"
    demand_range_color = "#C2B280"
    demand_avg_color = "#998066"

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    for ax_idx, (ax, node_name) in enumerate(zip(axs, node_names)):
        node_idx = np.where(node_names == node_name)[0][0]
        node_data = sim_outputs[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Boxplot styling
        boxprops = dict(linestyle='-', linewidth=1, alpha=0.9, facecolor=boxplot_color, edgecolor=boxplot_line_color)
        capprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        whiskerprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        medianprops = dict(linewidth=1, linestyle='-', alpha=0.9, color=boxplot_line_color)
        meanprops = dict(marker='o', markersize=9, markeredgecolor=boxplot_line_color, markerfacecolor=boxplot_line_color, alpha=0.9)

        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.5, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanprops)

        # Plot simulated mean with smoothing using interpolation
        sim_mean = np.mean(node_data, axis=0)
        ax.scatter(np.arange(0, 12), sim_mean, color=boxplot_color, facecolor=boxplot_line_color,
                   marker='o', alpha=0.9, s=10, zorder=4)
        interpolation_fn = interp1d(np.arange(1, 13), sim_mean, kind='cubic')
        smoothed_months = np.linspace(1, 12, num=100)
        sim_mean_smooth = interpolation_fn(smoothed_months)

        # Ensure smoothed values are not below 0
        sim_mean_smooth = np.maximum(sim_mean_smooth, 0)

        ax.plot(smoothed_months - 1, sim_mean_smooth, color=boxplot_color, linestyle='-', alpha=0.7, linewidth=2, zorder=1)

        # # Plot observed data with smoothing
        # obs = obs_data[node_idx]
        # ax.scatter(np.arange(0, 12), obs, color='red', edgecolor='gray', marker='o', alpha=0.9, s=5, zorder=4)
        # smo_obs = gaussian_filter1d(obs, sigma=0.1)
        # ax.plot(np.arange(0, 12), smo_obs, color='red', linestyle='-', alpha=0.5, linewidth=2, zorder=4)

        # Boxplot for demand range with specified color (only box, no whiskers)
        demand_range = np.vstack([demand_min[node_idx], demand_max[node_idx]])

        # Create a boxplot with customized parameters
        bp = ax.boxplot(demand_range, positions=np.arange(0, 12), widths=0.8,
                        patch_artist=True,  # Fill the boxes with color
                        showfliers=False,  # Hide outliers
                        showcaps=False,  # Hide caps
                        whiskerprops={'linewidth': 0},  # Set whisker linewidth to 0 to hide them
                        boxprops=dict(facecolor=demand_range_color, alpha=0.5),
                        medianprops=dict(color=demand_avg_color),
                        zorder=1)  # Set zorder to 1 to send it to the back

        # Iterate over the boxes and set the median line to be invisible
        for box in bp['boxes']:
            box.set(linewidth=0)  # Set box linewidth to 0 to hide the box outline
            box.set(facecolor=demand_range_color, alpha=0.5)  # Set fill color with transparency

        # Set the median line to be invisible
        for median in bp['medians']:
            median.set(linewidth=0)  # Set median linewidth to 0 to hide it

        # Set labels and titles
        ax.set_title(f'{node_name}')

        # Conditionally set x-label
        if ax_idx >= 4:  # Only set x-label for the second row
            ax.set_xlabel("Months")
        else:
            ax.set_xlabel("")

        month_labels = ["Oc", "Nv", "De", "Jn", "Fe", "Mr", "Ap", "My", "Jn", "Jl", "Au", "Se"]
        ax.set_xticklabels(month_labels)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)

        if ax_idx == 0 or ax_idx == 4:
            ax.set_ylabel("Water Volume (hm$^3$)")
            # Create legend for the first subplot with both elements as boxes
            if ax_idx == 0:  # Only add legend to the first subplot
                handles = [
                    mpatches.Patch(facecolor=boxplot_color, edgecolor='grey', label='Simulated Irrigation Water'),
                    mpatches.Patch(facecolor=demand_range_color, edgecolor='grey', label='Irrigation Demand Range')
                ]
                ax.legend(handles=handles, loc='upper left')
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_scenario(ax, data, smoothed_months, color, label):
    """
    Helper function to plot scenario data with cubic interpolation and smoothing.
    """
    if not np.isnan(data).all():
        interpolation_fn = interp1d(np.arange(1, 13), data, kind='cubic')
        smoothed_data = np.maximum(interpolation_fn(smoothed_months), 0)
        ax.plot(smoothed_months - 1, smoothed_data, color=color, linestyle='-', alpha=0.6, linewidth=1.5, label=label, zorder=5)
from matplotlib.lines import Line2D
def gemplot_irrigation_box(sim_outputs, obs_data, node_names, month_names, demand_min, demand_max,
                           SCE_BALANCED_TOT_IRG_WATER, SCE_MAXENRGY_TOT_IRG_WATER, SCE_SUSIRGAT_TOT_IRG_WATER, SCE_ECOCENTR_TOT_IRG_WATER):
    """
    Plots boxplots for irrigation water volumes and overlays scenario data.
    """
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)

    num_plots = len(node_names)
    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    # Color palette
    boxplot_color = "grey" #669966
    boxplot_line_color = "purple"
    demand_range_color = "#7DC462" #"#C2B280"
    demand_avg_color = "grey" #"#998066"

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    smoothed_months = np.linspace(1, 12, num=100)

    for ax_idx, (ax, node_name) in enumerate(zip(axs, node_names)):
        node_idx = np.where(node_names == node_name)[0][0]
        node_data = sim_outputs[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Boxplot styling
        boxprops = dict(linestyle='-', linewidth=1, alpha=0.9, facecolor='none', edgecolor=boxplot_line_color)
        capprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        whiskerprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        medianprops = dict(linewidth=1, linestyle='-', alpha=0.9, color=boxplot_line_color)
        meanprops = dict(marker='o', markersize=9, markeredgecolor=boxplot_line_color, markerfacecolor=boxplot_line_color, alpha=0.9)

        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.8, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanprops)

        # Plot simulated mean with smoothing using interpolation
        sim_mean = np.mean(node_data, axis=0)
        # ax.scatter(np.arange(0, 12), sim_mean, color=boxplot_color, facecolor=boxplot_line_color,
        #            marker='o', alpha=0.9, s=10, zorder=4)

        # Plot demand range as a boxplot without whiskers
        demand_range = np.vstack([demand_min[node_idx], demand_max[node_idx]])
        bp = ax.boxplot(demand_range, positions=np.arange(0, 12), widths=0.8,
                        patch_artist=True,  # Fill the boxes with color
                        showfliers=False,  # Hide outliers
                        showcaps=False,  # Hide caps
                        whiskerprops={'linewidth': 0},  # Hide whiskers
                        boxprops=dict(facecolor=demand_range_color, alpha=0.5),
                        medianprops=dict(color=demand_avg_color),
                        zorder=1)

        # Iterate over the boxes and set the median line to be invisible
        for box in bp['boxes']:
            box.set(linewidth=0)  # Set box linewidth to 0 to hide the box outline
            box.set(facecolor=demand_range_color, alpha=0.5)  # Set fill color with transparency

        # Set the median line to be invisible
        for median in bp['medians']:
            median.set(linewidth=0)  # Set median linewidth to 0 to hide it

        # Plot scenarios
        # plot_scenario(ax, SCE_BALANCED_TOT_IRG_WATER[node_idx], smoothed_months, "purple", "Balanced")
        # plot_scenario(ax, SCE_MAXENRGY_TOT_IRG_WATER[node_idx], smoothed_months, "orange", "Maximizing Energy")
        # ax.plot(np.arange(0, 12), SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], color="#008080", linestyle='-', alpha=0.9, linewidth=1.5, label="Sustainable Irrigation", zorder=5)
        ax.scatter(np.arange(0, 12), SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], facecolor="#008080", edgecolor="black",
                   marker='o', alpha=0.9, s=20, zorder=4)
        plot_scenario(ax, SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], smoothed_months, "#008080", "Sustainable Irrigation")
        # plot_scenario(ax, SCE_ECOCENTR_TOT_IRG_WATER[node_idx], smoothed_months, "#00BFFF", "Eco Centric")

        # Set title and axis labels
        ax.set_title(f'{node_name}', pad=10, loc='center')
        if ax_idx >= 4:  # Only set x-label for the second row
            ax.set_xlabel("Months")
        else:
            ax.set_xlabel("")

        month_labels = ["Oc", "Nv", "De", "Jn", "Fe", "Mr", "Ap", "My", "Jn", "Jl", "Au", "Se"]
        ax.set_xticklabels(month_labels)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)

        if ax_idx == 0 or ax_idx == 4:
            ax.set_ylabel("Water Volume (hm$^3$)")
            if ax_idx == 0:  # Add legend to the first subplot
                handles = [
                    mpatches.Patch(facecolor="none", edgecolor='purple', label='Simulated Range'),
                    mpatches.Patch(facecolor=demand_range_color, edgecolor='grey', label='Demand Range'),
                    Line2D([0], [0], color="#008080", lw=2, label="Sustainable Irrigation"),
                ]
                ax.legend(handles=handles, loc='upper left')
        else:
            ax.set_ylabel("")


    plt.tight_layout()
    plt.show()

def thousands_formatter_turkish(x, pos):
    """
    Sayıyı binlik ayıracı olarak nokta (.) ve ondalık ayıracı olarak virgül (,)
    kullanarak Türkçe formatına dönüştürür.
    """
    s = f'{x:,.1f}'  # always with 2 decimals
    s = s.replace(',', 'T')   # temp replace thousands comma
    s = s.replace('.', ',')   # replace decimal dot with comma
    s = s.replace('T', '.')   # restore thousands separator with dot
    return s

def gemplot_irrigation_box_turkish_single(sim_outputs, obs_data, node_names, month_names, demand_min, demand_max,
                           SCE_BALANCED_TOT_IRG_WATER, SCE_MAXENRGY_TOT_IRG_WATER, SCE_SUSIRGAT_TOT_IRG_WATER, SCE_ECOCENTR_TOT_IRG_WATER):
    """
    Plots separate boxplots for irrigation water volumes and overlays scenario data.
    Each node will have its own figure.
    """
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)

    # Color palette
    boxplot_color = "grey"
    boxplot_line_color = "grey"
    demand_range_color = "#DFD7BF"
    demand_avg_color = "grey"

    smoothed_months = np.linspace(1, 12, num=100)

    for node_idx, node_name in enumerate(node_names):
        # Create new figure for each node
        fig, ax = plt.subplots(figsize=(8.5, 5))

        node_data = sim_outputs[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Boxplot styling
        boxprops = dict(linestyle='-', linewidth=1, alpha=0.9, facecolor='none', edgecolor=boxplot_line_color)
        capprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        whiskerprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        medianprops = dict(linewidth=1, linestyle='-', alpha=0.9, color=boxplot_line_color)
        meanprops = dict(marker='o', markersize=9, markeredgecolor=boxplot_line_color, markerfacecolor=boxplot_line_color, alpha=0.9)

        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.8, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanprops)

        # Plot demand range as filled area
        demand_range = np.vstack([demand_min[node_idx], demand_max[node_idx]])
        bp = ax.boxplot(demand_range, positions=np.arange(0, 12), widths=0.8,
                        patch_artist=True, showfliers=False, showcaps=False,
                        whiskerprops={'linewidth': 0},
                        boxprops=dict(facecolor=demand_range_color, alpha=0.8),
                        medianprops=dict(color=demand_avg_color),
                        zorder=1)

        # Hide outlines
        for box in bp['boxes']:
            box.set(linewidth=0)
        for median in bp['medians']:
            median.set(linewidth=0)

        # Plot scenario example
        ax.scatter(np.arange(0, 12), SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], facecolor="#008080", edgecolor="white",
                   marker='o', alpha=0.9, s=40, zorder=4)
        plot_scenario(ax, SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], smoothed_months, "#008080", "Sustainable Irrigation")

        # Labels
        ax.set_title(f'{node_name}', pad=10, loc='center')
        ax.set_xlabel("Aylar")
        ax.set_ylabel("Sulama Suyu (hm$^3$)")
        ax.set_xticklabels(["Ekim", "Kasım", "Aralık", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül"])
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_turkish))

        # Optional legend only on first plot
        if node_idx == 0:
            handles = [
                mpatches.Patch(facecolor="none", edgecolor='purple', label='Simulated Range'),
                mpatches.Patch(facecolor=demand_range_color, edgecolor='grey', label='Demand Range'),
                Line2D([0], [0], color="#008080", lw=2, label="Sustainable Irrigation"),
            ]
            # ax.legend(handles=handles, loc='upper left')

        plt.tight_layout()
        plt.show()

def gemplot_irrigation_box_turkish(sim_outputs, obs_data, node_names, month_names, demand_min, demand_max,
                           SCE_BALANCED_TOT_IRG_WATER, SCE_MAXENRGY_TOT_IRG_WATER, SCE_SUSIRGAT_TOT_IRG_WATER, SCE_ECOCENTR_TOT_IRG_WATER):
    """
    Plots boxplots for irrigation water volumes and overlays scenario data.
    """
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)

    num_plots = len(node_names)
    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    # Color palette
    boxplot_color = "grey" #669966
    boxplot_line_color = "grey"
    demand_range_color = "#DFD7BF" #"#C2B280"
    demand_avg_color = "grey" #"#998066"


    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    smoothed_months = np.linspace(1, 12, num=100)

    for ax_idx, (ax, node_name) in enumerate(zip(axs, node_names)):
        node_idx = np.where(node_names == node_name)[0][0]
        node_data = sim_outputs[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Boxplot styling
        boxprops = dict(linestyle='-', linewidth=1, alpha=0.9, facecolor='none', edgecolor=boxplot_line_color)
        capprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        whiskerprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        medianprops = dict(linewidth=1, linestyle='-', alpha=0.9, color=boxplot_line_color)
        meanprops = dict(marker='o', markersize=9, markeredgecolor=boxplot_line_color, markerfacecolor=boxplot_line_color, alpha=0.9)

        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.8, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanprops)

        # Plot simulated mean with smoothing using interpolation
        sim_mean = np.mean(node_data, axis=0)
        # ax.scatter(np.arange(0, 12), sim_mean, color=boxplot_color, facecolor=boxplot_line_color,
        #            marker='o', alpha=0.9, s=10, zorder=4)

        # Plot demand range as a boxplot without whiskers
        demand_range = np.vstack([demand_min[node_idx], demand_max[node_idx]])
        bp = ax.boxplot(demand_range, positions=np.arange(0, 12), widths=0.8,
                        patch_artist=True,  # Fill the boxes with color
                        showfliers=False,  # Hide outliers
                        showcaps=False,  # Hide caps
                        whiskerprops={'linewidth': 0},  # Hide whiskers
                        boxprops=dict(facecolor=demand_range_color, alpha=0.8),
                        medianprops=dict(color=demand_avg_color),
                        zorder=1)

        # Iterate over the boxes and set the median line to be invisible
        for box in bp['boxes']:
            box.set(linewidth=0)  # Set box linewidth to 0 to hide the box outline
            box.set(facecolor=demand_range_color, alpha=0.8)  # Set fill color with transparency

        # Set the median line to be invisible
        for median in bp['medians']:
            median.set(linewidth=0)  # Set median linewidth to 0 to hide it

        # Plot scenarios
        # plot_scenario(ax, SCE_BALANCED_TOT_IRG_WATER[node_idx], smoothed_months, "purple", "Balanced")
        # plot_scenario(ax, SCE_MAXENRGY_TOT_IRG_WATER[node_idx], smoothed_months, "orange", "Maximizing Energy")
        # ax.plot(np.arange(0, 12), SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], color="#008080", linestyle='-', alpha=0.9, linewidth=1.5, label="Sustainable Irrigation", zorder=5)
        ax.scatter(np.arange(0, 12), SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], facecolor="#008080", edgecolor="white",
                   marker='o', alpha=0.9, s=25, zorder=4)
        plot_scenario(ax, SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], smoothed_months, "#008080", "Sustainable Irrigation")
        # plot_scenario(ax, SCE_ECOCENTR_TOT_IRG_WATER[node_idx], smoothed_months, "#00BFFF", "Eco Centric")

        # Set title and axis labels
        ax.set_title(f'{node_name}', pad=10, loc='center')
        if ax_idx >= 4:  # Only set x-label for the second row
            ax.set_xlabel("Aylar")
        else:
            ax.set_xlabel("")

        month_labels =  ["Ek", "Ks", "Ar", "Oc", "Şb", "Mr", "Ns", "My", "Hz", "Tm", "Ağ", "Ey"]
        ax.set_xticklabels(month_labels)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_turkish))

        if ax_idx == 0 or ax_idx == 4:
            ax.set_ylabel("Sulama Suyu (hm$^3$)")
            if ax_idx == 0:  # Add legend to the first subplot
                handles = [
                    mpatches.Patch(facecolor="none", edgecolor='purple', label='Simulated Range'),
                    mpatches.Patch(facecolor=demand_range_color, edgecolor='grey', label='Demand Range'),
                    Line2D([0], [0], color="#008080", lw=2, label="Sustainable Irrigation"),
                ]
                # ax.legend(handles=handles, loc='upper left')
        else:
            ax.set_ylabel("")


    plt.tight_layout()
    plt.show()

def plot_irrigation_individually_turkish(sim_outputs, obs_data, node_names, month_names, demand_min, demand_max,
                                         SCE_BALANCED_TOT_IRG_WATER, SCE_MAXENRGY_TOT_IRG_WATER, SCE_SUSIRGAT_TOT_IRG_WATER,
                                         SCE_ECOCENTR_TOT_IRG_WATER):
    """
    Generates and displays individual figures for each irrigation node.
    """
    flierprops = dict(marker='o', markersize=1, linestyle='none', alpha=0)

    # Color palette
    boxplot_line_color = "grey"
    demand_range_color = "#DFD7BF"
    demand_avg_color = "grey"

    # Flattening is not necessary here, but we need to iterate over nodes
    num_nodes = len(node_names)

    # X-axis labels (Turkish 2-char)
    TR_MONTHS_2CHAR = ["Ekim", "Kasım", "Aralık", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül"]

    # --- NODE LOOP STARTS HERE (Each iteration creates one figure) ---
    for node_idx, node_name in enumerate(node_names):

        # 🔥 CRITICAL CHANGE: Create a new figure and single subplot for each node 🔥
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed for individual plots

        node_data = sim_outputs[:, node_idx, :]

        # Convert to long format for seaborn
        df = pd.DataFrame(node_data, columns=[f'{i + 1}' for i in range(12)])
        df = df.melt(var_name="Month", value_name="Water", ignore_index=False).reset_index()

        # Boxplot styling
        boxprops = dict(linestyle='-', linewidth=1, alpha=0.9, facecolor='none', edgecolor=boxplot_line_color)
        capprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        whiskerprops = dict(linewidth=1, alpha=0.7, color=boxplot_line_color)
        medianprops = dict(linewidth=1, linestyle='-', alpha=0.9, color=boxplot_line_color)
        meanprops = dict(marker='o', markersize=9, markeredgecolor=boxplot_line_color, markerfacecolor=boxplot_line_color, alpha=0.9)

        sns.boxplot(x='Month', y='Water', data=df, ax=ax,
                    width=0.8, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanprops, showmeans=False)

        # --- TALEP ARALIĞI KUTU GRAFİKLERİ ---
        demand_range = np.vstack([demand_min[node_idx], demand_max[node_idx]])
        month_indices = np.arange(0, 12)

        bp = ax.boxplot(demand_range, positions=month_indices, widths=0.8,
                        patch_artist=True,
                        showfliers=False,
                        showcaps=False,
                        whiskerprops={'linewidth': 0},
                        boxprops=dict(facecolor=demand_range_color, alpha=0.8),
                        medianprops=dict(color=demand_avg_color),
                        zorder=1)

        # Style demand boxes
        for box in bp['boxes']:
            box.set(linewidth=0)
            box.set(facecolor=demand_range_color, alpha=0.8)
        for median in bp['medians']:
            median.set(linewidth=0)  # Hide median line

        # --- SENARYO ÇİZİMLERİ ---
        sim_mean = np.mean(node_data, axis=0)
        ax.scatter(month_indices, sim_mean, color=boxplot_line_color, marker='o', alpha=0.9, s=20, zorder=4)

        # Sustainable Irrigation Scatter and Line
        ax.scatter(month_indices, SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], facecolor="#008080", edgecolor="black",
                   marker='o', alpha=0.9, s=30, zorder=6, label="Sürdürülebilir Sulama Ortalaması")
        plot_scenario(ax, SCE_SUSIRGAT_TOT_IRG_WATER[node_idx], None, "#008080", "Sürdürülebilir Sulama Çizgisi")


        # --- FİNAL STİL VE EKSENLER ---
        ax.set_title(f'Sulama Suyu {node_name}', pad=10, loc='center')
        ax.set_xlabel("Aylar", fontsize=10)

        # Y-Ekseni Etiketleri (Her grafikte gösterilir)
        ax.set_ylabel("Sulama Suyu (hm$^3$)", fontsize=10)

        ax.set_xticklabels(TR_MONTHS_2CHAR, fontsize=9)
        ax.set_xticks(month_indices)

        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter_turkish))
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)

        # Lejant (Her grafiğe ayrı ayrı)
        handles = [
            mpatches.Patch(facecolor="none", edgecolor=boxplot_line_color, label='Simülasyon Aralığı'),
            mpatches.Patch(facecolor=demand_range_color, edgecolor=demand_avg_color, label='Talep Aralığı'),
            Line2D([0], [0], color="#008080", lw=2, label="Sürdürülebilir Sulama"),
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=8)

        # Display and close the individual figure
        plt.tight_layout()
        plt.show()
        plt.close(fig)  #


def plot_boxplot_outflow(SIM_OUTPUT, OBS_DATA, TARGET_DATA, NODE_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None, debug_mode=False,
                         debug_idx=0):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)
    arrowprops = dict(color='dimgray', arrowstyle="->", lw=1.8, shrinkA=5, shrinkB=5,
                      connectionstyle="arc3,rad=.0", alpha=0.8, zorder=4)

    # Use the same color palette as gemplot_env_management_flow
    scenario_colors = ["darkgrey", "#74c476", "purple"]

    for node_idx, node_name in enumerate(NODE_NAMES):
        node_data = SIM_OUTPUT[:, node_idx, :]

        if debug_mode and node_idx != debug_idx:
            continue

        plt.figure(figsize=(10, 6))

        # --- Smooth the observed data using interpolation ---
        obs = OBS_DATA[node_idx, :]
        trg = TARGET_DATA[node_idx, :]
        sim = np.mean(node_data, axis=0)

        interpolation_obs = interp1d(np.arange(1, 13), obs, kind='cubic')
        interpolation_trg = interp1d(np.arange(1, 13), trg, kind='cubic')
        interpolation_sim = interp1d(np.arange(1, 13), sim, kind='cubic')

        smoothed_months = np.linspace(1, 12, 90)

        obs_smooth = interpolation_obs(smoothed_months)
        trg_smooth = interpolation_trg(smoothed_months)
        sim_smooth = interpolation_sim(smoothed_months)

        # Plot the smoothed observed data
        plt.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        plt.scatter(np.arange(1, 13), obs, edgecolors=scenario_colors[0], facecolors="white", marker='o', alpha=0.9, s=50, label="Observed Mean",
                    zorder=4)

        # Plot target mean
        plt.plot(smoothed_months, trg_smooth, color=scenario_colors[1], alpha=1, zorder=1)
        plt.scatter(np.arange(1, 13), trg, edgecolors=scenario_colors[1], facecolors="white", marker='o', alpha=0.9, s=50, label="Env. Mng. Class B",
                    zorder=2)

        # Plot simulated mean
        plt.plot(smoothed_months, sim_smooth, color=scenario_colors[2], alpha=0.8, zorder=1)
        # plt.plot(np.arange(1, 13), sim, color=scenario_colors[-2], alpha=0.8)
        plt.scatter(np.arange(1, 13), sim, edgecolors=scenario_colors[2], facecolors="white", marker='o', alpha=0.9, s=50, label="Simulated Mean",
                    zorder=4)

        # --- Calculate and smooth the min-max range ---
        sim_min = np.min(node_data, axis=0)
        sim_max = np.max(node_data, axis=0)

        interpolation_min = interp1d(np.arange(1, 13), sim_min, kind='cubic')
        interpolation_max = interp1d(np.arange(1, 13), sim_max, kind='cubic')

        sim_min_smooth = interpolation_min(smoothed_months)
        sim_max_smooth = interpolation_max(smoothed_months)

        # Fill between min-max range
        plt.fill_between(smoothed_months, sim_min_smooth, sim_max_smooth, color=scenario_colors[2], alpha=0.1)
        # plt.fill_between(np.arange(1, 13), sim_min, sim_max, color=scenario_colors[-2], alpha=0.1)

        #
        # # Simulated Distribution
        # for month_idx in range(12):
        #     month_val = node_data[:, month_idx]
        #     x_val = np.repeat(month_idx + 1, len(month_val))
        #     plt.scatter(x_val, month_val, color="silver", alpha=0.1)

        # Add arrows from observed mean to simulated mean
        arrowprops = dict(color="#222222", arrowstyle="simple", lw=1, shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=.0", alpha=0.7, zorder=5)
        for i, (obs_val, sim_val) in enumerate(zip(obs, sim), start=1):
            plt.annotate('', xy=(i, sim_val), xytext=(i, obs_val), arrowprops=arrowprops)

        plt.title(f'{node_name}')
        # plt.xlabel('Months')
        plt.ylabel('Streamflow (hm³)')
        plt.xticks(ticks=np.arange(1, 13), labels=CH_MONTHS_NAME)
        plt.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.show()

def plot_boxplot_outflow_scatter(SIM_OUTPUT, OBS_DATA, TARGET_DATA, NODE_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None, debug_mode=False,
                         debug_idx=0):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)
    arrowprops = dict(color='dimgray', arrowstyle="->", lw=1.8, shrinkA=5, shrinkB=5,
                      connectionstyle="arc3,rad=.0", alpha=0.8, zorder=4)

    # Use the same color palette as gemplot_env_management_flow
    scenario_colors = ["#D3D9DB", "#74c476", "purple"]

    for node_idx, node_name in enumerate(NODE_NAMES):
        node_data = SIM_OUTPUT[:, node_idx, :]

        if debug_mode and node_idx != debug_idx:
            continue

        plt.figure(figsize=(12, 8), dpi=200)

        # --- Smooth the observed data using interpolation ---
        obs = OBS_DATA[node_idx, :]
        trg = TARGET_DATA[node_idx, :]
        sim = np.mean(node_data, axis=0)

        # interpolation_obs = interp1d(np.arange(1, 13), obs, kind='cubic')
        # interpolation_trg = interp1d(np.arange(1, 13), trg, kind='cubic')
        # interpolation_sim = interp1d(np.arange(1, 13), sim, kind='cubic')
        #
        # smoothed_months = np.linspace(1, 12, 90)
        #
        # obs_smooth = interpolation_obs(smoothed_months)
        # trg_smooth = interpolation_trg(smoothed_months)
        # sim_smooth = interpolation_sim(smoothed_months)

        # Plot the smoothed observed data
        # plt.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        plt.scatter(np.arange(1, 13), obs, edgecolors="#96999A", facecolors=scenario_colors[0], marker='o', alpha=0.9, s=50, label="Reference Mean",
                    zorder=4)

        # Plot target mean
        # plt.plot(smoothed_months, trg_smooth, color=scenario_colors[1], alpha=1, zorder=1)
        plt.scatter(np.arange(1, 13), trg, edgecolors="#69A953", facecolors=scenario_colors[1], marker='o', alpha=0.9, s=50, label="Env. Mng. Class B",
                    zorder=2)

        # Plot simulated mean
        # plt.plot(smoothed_months, sim_smooth, color=scenario_colors[2], alpha=0.8, zorder=1)
        # plt.plot(np.arange(1, 13), sim, color=scenario_colors[-2], alpha=0.8)
        plt.scatter(np.arange(1, 13), sim, edgecolors="#563973", facecolors=scenario_colors[2], marker='o', alpha=0.9, s=50, label="Simulated Mean",
                    zorder=4)

        # --- Calculate and smooth the min-max range ---
        # sim_min = np.min(node_data, axis=0)
        # sim_max = np.max(node_data, axis=0)
        #
        # interpolation_min = interp1d(np.arange(1, 13), sim_min, kind='cubic')
        # interpolation_max = interp1d(np.arange(1, 13), sim_max, kind='cubic')
        #
        # sim_min_smooth = interpolation_min(smoothed_months)
        # sim_max_smooth = interpolation_max(smoothed_months)

        # Fill between min-max range
        # plt.fill_between(smoothed_months, sim_min_smooth, sim_max_smooth, color=scenario_colors[2], alpha=0.1)
        # plt.fill_between(np.arange(1, 13), sim_min, sim_max, color=scenario_colors[-2], alpha=0.1)

        #
        # Simulated Distribution
        for month_idx in range(12):
            month_val = node_data[:, month_idx]
            x_val = np.repeat(month_idx + 1, len(month_val))
            plt.scatter(x_val, month_val, edgecolors="#BBAAD3", facecolors="none", alpha=0.1)

        # Add arrows from observed mean to simulated mean
        # arrowprops = dict(color="#222222", arrowstyle="simple", lw=1, shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=.0", alpha=0.7, zorder=5)
        # for i, (obs_val, sim_val) in enumerate(zip(obs, sim), start=1):
        #     plt.annotate('', xy=(i, sim_val), xytext=(i, obs_val), arrowprops=arrowprops)

        plt.title(f'{node_name}')
        # plt.xlabel('Months')
        plt.ylabel('Streamflow (hm³)')
        plt.xticks(ticks=np.arange(1, 13), labels=CH_MONTHS_NAME)
        plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
        plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.show()

def plot_boxplot_outflow_scatter_turkish(SIM_OUTPUT, OBS_DATA, TARGET_DATA, NODE_NAMES, LOWER_LIMIT=None, UPPER_LIMIT=None, INITIAL=None, debug_mode=False,
                         debug_idx=0):
    boxprops = dict(linestyle='-', linewidth=1.5, color='gray', facecolor="whitesmoke", alpha=0.7)
    flierprops = dict(marker='o', markersize=1.5, linestyle='none', color='none', alpha=0)
    whiskerprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    capprops = dict(color='gray', linewidth=1.5, alpha=0.7)
    medianprops = dict(linewidth=1.5, linestyle='-', color='gray', alpha=0.7)
    meanprops = dict(marker='o', markersize=9, markeredgecolor='maroon', markerfacecolor='crimson', alpha=0.9)
    arrowprops = dict(color='dimgray', arrowstyle="->", lw=1.8, shrinkA=5, shrinkB=5,
                      connectionstyle="arc3,rad=.0", alpha=0.8, zorder=4)

    # Use the same color palette as gemplot_env_management_flow
    scenario_colors = ["#D3D9DB", "#74c476", "purple"]

    for node_idx, node_name in enumerate(NODE_NAMES):
        node_data = SIM_OUTPUT[:, node_idx, :]

        if debug_mode and node_idx != debug_idx:
            continue

        plt.figure(figsize=(8, 4))

        # --- Smooth the observed data using interpolation ---
        obs = OBS_DATA[node_idx, :]
        trg = TARGET_DATA[node_idx, :]
        sim = np.mean(node_data, axis=0)

        # interpolation_obs = interp1d(np.arange(1, 13), obs, kind='cubic')
        # interpolation_trg = interp1d(np.arange(1, 13), trg, kind='cubic')
        # interpolation_sim = interp1d(np.arange(1, 13), sim, kind='cubic')
        #
        # smoothed_months = np.linspace(1, 12, 90)
        #
        # obs_smooth = interpolation_obs(smoothed_months)
        # trg_smooth = interpolation_trg(smoothed_months)
        # sim_smooth = interpolation_sim(smoothed_months)

        # Plot the smoothed observed data
        # plt.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        plt.scatter(np.arange(1, 13), obs, edgecolors="#96999A", facecolors=scenario_colors[0], marker='o', alpha=0.9, s=50, label="Ortalama Mevcut Akım",
                    zorder=4)

        # Plot target mean
        # plt.plot(smoothed_months, trg_smooth, color=scenario_colors[1], alpha=1, zorder=1)
        plt.scatter(np.arange(1, 13), trg, edgecolors="#69A953", facecolors=scenario_colors[1], marker='o', alpha=0.9, s=50, label="Referans Akım",
                    zorder=2)

        # Plot simulated mean
        # plt.plot(smoothed_months, sim_smooth, color=scenario_colors[2], alpha=0.8, zorder=1)
        # plt.plot(np.arange(1, 13), sim, color=scenario_colors[-2], alpha=0.8)
        plt.scatter(np.arange(1, 13), sim, edgecolors="#563973", facecolors=scenario_colors[2], marker='o', alpha=0.9, s=50, label="Simülasyon Ortalama",
                    zorder=4)

        # --- Calculate and smooth the min-max range ---
        # sim_min = np.min(node_data, axis=0)
        # sim_max = np.max(node_data, axis=0)
        #
        # interpolation_min = interp1d(np.arange(1, 13), sim_min, kind='cubic')
        # interpolation_max = interp1d(np.arange(1, 13), sim_max, kind='cubic')
        #
        # sim_min_smooth = interpolation_min(smoothed_months)
        # sim_max_smooth = interpolation_max(smoothed_months)

        # Fill between min-max range
        # plt.fill_between(smoothed_months, sim_min_smooth, sim_max_smooth, color=scenario_colors[2], alpha=0.1)
        # plt.fill_between(np.arange(1, 13), sim_min, sim_max, color=scenario_colors[-2], alpha=0.1)

        #
        # Simulated Distribution
        for month_idx in range(12):
            month_val = node_data[:, month_idx]
            x_val = np.repeat(month_idx + 1, len(month_val))
            plt.scatter(x_val, month_val, edgecolors="#BBAAD3", facecolors="none", alpha=0.1)

        # Add arrows from observed mean to simulated mean
        # arrowprops = dict(color="#222222", arrowstyle="simple", lw=1, shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=.0", alpha=0.7, zorder=5)
        # for i, (obs_val, sim_val) in enumerate(zip(obs, sim), start=1):
        #     plt.annotate('', xy=(i, sim_val), xytext=(i, obs_val), arrowprops=arrowprops)

        plt.title(f'{node_name}')
        # plt.xlabel('Months')
        plt.ylabel('Akım (hm³)')
        AYLAR = ["Eki", "Kas", "Ara", "Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl"]
        plt.xticks(ticks=np.arange(1, 13), labels=AYLAR)
        plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
        # plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.show()


def gemplot_env_management_flow(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES):
    scenario_colors = ["darkgray", "#045275", "#089099", "#7ccba2", "#f0746e", "#dc3977", "#7c1d6f"]
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    bar_width = 0.15  # Width of each bar

    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    axs = axs.flatten()

    # Determine the overall y-axis limits with padding
    y_min = 0
    y_max = 400

    for node_index, node_name in enumerate(NODE_NAMES):
        ax = axs[node_index]

        # --- Plot observed data ---
        obs = OBS_DATA[node_index, :]
        ax.bar(np.arange(1, 13) - 1 * bar_width, obs, width=bar_width, color=scenario_colors[0], label=scenario_labels[0], zorder=5)

        # Plot simulated data as bar chart
        for i, sim_output in enumerate(SIM_OUTPUT_LIST):
            node_data = sim_output[node_index, :]
            ax.bar(np.arange(1, 13) + i * bar_width, node_data, width=bar_width, color=scenario_colors[i + 1], label=scenario_labels[i + 1])

        ax.set_title(f"{node_name}")

        # X-axis labels and ticks
        if node_index >= 4:
            ax.set_xticks(np.arange(1, 13))
            ax.set_xticklabels(month_labels)
        else:
            ax.set_xticks([])

        # Y-axis label
        if node_index == 0 or node_index == 4:
            ax.set_ylabel('Streamflow (hm³)')
        else:
            ax.set_ylabel("")

        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis='y', linestyle='-', linewidth=0.5, color='lightgrey')

        # Legend in the first subplot
        if node_index == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    plt.show()


def line_gemplot_env_management_flow(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES):
    scenario_colors = ["darkgray", "#007A1F", "#74c476", "#FFA500", "#fdae61", "#f46d43", "#d73027"]
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    axs = axs.flatten()

    for node_index, node_name in enumerate(NODE_NAMES):
        ax = axs[node_index]

        smoothed_months = np.linspace(1, 12, 90)

        obs = OBS_DATA[node_index, :]
        interpolation_fn = interp1d(np.arange(1, 13), obs, kind='cubic')
        obs_smooth = interpolation_fn(smoothed_months)

        ax.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=5)
        ax.scatter(np.arange(1, 13), obs, edgecolor=scenario_colors[0], facecolor="white", alpha=0.9,  label=scenario_labels[0], zorder=6, s=20)

        # Plot simulated data with smoothed lines
        for i, sim_output in enumerate(SIM_OUTPUT_LIST):
            node_data = sim_output[node_index, :]
            interpolation_fn = interp1d(np.arange(1, 13), node_data, kind='cubic')
            smoothed_outflow = interpolation_fn(smoothed_months)

            ax.plot(smoothed_months, smoothed_outflow, color=scenario_colors[i + 1], linewidth=1.2, alpha=0.6, zorder=3)
            ax.scatter(np.arange(1, 13), node_data, color=scenario_colors[i + 1], alpha=0.8, label=scenario_labels[i + 1], s=15, zorder=4)

        ax.set_title(f"{node_name}")
        ax.grid(True, linewidth=0.5, color='lightgrey')  # Added grid to all subplots

        # X-axis labels and ticks (corrected)
        ax.set_xticks(np.arange(1, 13))  # Set x-ticks for all subplots
        ax.set_xticklabels(month_labels)

        # Y-axis label
        if node_index == 0 or node_index == 4:
            ax.set_ylabel('Streamflow (hm³)')
        else:
            ax.set_ylabel("")

        # Legend in the first subplot (moved to upper left)
        if node_index == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))  # Adjusted position

    plt.tight_layout()
    plt.show()

def line_gemplot_env_management_flow_turkish(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES):
    scenario_colors = ["grey", "#007A1F", "#8BC55C", "#FFA500", "#F28E2B", "#f46d43", "#d73027"] #yeniledim
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Eki", "Kas", "Ara", "Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl"]

    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    axs = axs.flatten()

    for node_index, node_name in enumerate(NODE_NAMES):
        ax = axs[node_index]

        smoothed_months = np.linspace(1, 12, 90)

        obs = OBS_DATA[node_index, :]
        interpolation_fn = interp1d(np.arange(1, 13), obs, kind='cubic')
        obs_smooth = interpolation_fn(smoothed_months)
        obs_smooth = np.maximum(obs_smooth, 0)

        ax.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=5)
        ax.scatter(np.arange(1, 13), obs, edgecolor=scenario_colors[0], facecolor="white", alpha=0.9,  label=scenario_labels[0], zorder=6, s=20)

        # Plot simulated data with smoothed lines
        for i, sim_output in enumerate(SIM_OUTPUT_LIST):
            node_data = sim_output[node_index, :]
            interpolation_fn = interp1d(np.arange(1, 13), node_data, kind='cubic')
            smoothed_outflow = interpolation_fn(smoothed_months)

            # 🔥 DÜZELTME 2: Negatif akım değerlerini sıfıra indir
            smoothed_outflow = np.maximum(smoothed_outflow, 0)

            ax.plot(smoothed_months, smoothed_outflow, color=scenario_colors[i + 1], linewidth=1.2, alpha=0.6, zorder=3)
            ax.scatter(np.arange(1, 13), node_data, color=scenario_colors[i + 1], alpha=0.8, label=scenario_labels[i + 1], s=15, zorder=4)

        ax.set_title(f"{node_name}", pad=15)
        ax.grid(True, linewidth=0.5, color='lightgrey')  # Added grid to all subplots

        # X-axis labels and ticks (corrected)
        ax.set_xticks(np.arange(1, 13))  # Set x-ticks for all subplots
        ax.set_xticklabels(month_labels)

        # Y-axis label
        if node_index == 0 or node_index == 4:
            ax.set_ylabel('Akım (hm³)')
        else:
            ax.set_ylabel("")

        # # Legend in the first subplot (moved to upper left)
        # if node_index == 0:
        #     ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))  # Adjusted position

    plt.tight_layout()
    plt.show()

def line_gemplot_env_management_flow_single(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES, node_index):
    scenario_colors = ["#0077be", "#007A1F", "#74c476", "#FFA500", "#fdae61", "#f46d43", "#d73027"]
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    smoothed_months = np.linspace(1, 12, 90)

    obs = OBS_DATA[node_index, :]
    interpolation_fn = interp1d(np.arange(1, 13), obs, kind='cubic')
    obs_smooth = interpolation_fn(smoothed_months)

    ax.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=5)
    ax.scatter(np.arange(1, 13), obs, edgecolor=scenario_colors[0], facecolor="white", label=scenario_labels[0], zorder=6, s=20)

    for i, sim_output in enumerate(SIM_OUTPUT_LIST):
        node_data = sim_output[node_index, :]
        interpolation_fn = interp1d(np.arange(1, 13), node_data, kind='cubic')
        smoothed_outflow = interpolation_fn(smoothed_months)

        ax.plot(smoothed_months, smoothed_outflow, color=scenario_colors[i + 1], linewidth=1.2, alpha=0.8, zorder=3)
        ax.scatter(np.arange(1, 13), node_data, color=scenario_colors[i + 1], alpha=0.9, label=scenario_labels[i + 1], s=15, zorder=4)

    ax.set_title(f"{NODE_NAMES[node_index]}")
    ax.grid(True, linewidth=0.5, color='lightgrey')

    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(month_labels)

    ax.set_ylabel('Streamflow (hm³)')

    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.show()

def line_gemplot_env_management_flow_single_turkish(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES, node_index):
    scenario_colors = ["grey", "#007A1F", "#8BC55C", "#FFA500", "#F28E2B", "#f46d43", "#d73027"] #yeniledim
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Eki", "Kas", "Ara", "Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl"]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    smoothed_months = np.linspace(1, 12, 90)

    obs = OBS_DATA[node_index, :]
    interpolation_fn = interp1d(np.arange(1, 13), obs, kind='cubic')
    obs_smooth = interpolation_fn(smoothed_months)
    obs_smooth = np.maximum(obs_smooth, 0)

    ax.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=5)
    ax.scatter(np.arange(1, 13), obs, edgecolor=scenario_colors[0], facecolor="white", label=scenario_labels[0], zorder=6, s=20)

    for i, sim_output in enumerate(SIM_OUTPUT_LIST):
        node_data = sim_output[node_index, :]
        interpolation_fn = interp1d(np.arange(1, 13), node_data, kind='cubic')
        smoothed_outflow = interpolation_fn(smoothed_months)

        # 🔥 DÜZELTME 2: Negatif akım değerlerini sıfıra indir
        smoothed_outflow = np.maximum(smoothed_outflow, 0)

        ax.plot(smoothed_months, smoothed_outflow, color=scenario_colors[i + 1], linewidth=1.2, alpha=0.8, zorder=3)
        ax.scatter(np.arange(1, 13), node_data, color=scenario_colors[i + 1], alpha=0.9, label=scenario_labels[i + 1], s=15, zorder=4)

    ax.set_title(f"{NODE_NAMES[node_index]}", fontsize=16, pad=15)

    ax.grid(True, linewidth=0.5, color='lightgrey')

    ax.set_xticks(np.arange(1, 13))

    ax.set_xticklabels(month_labels, fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.set_xlabel("Aylar", fontsize=14)
    ax.set_ylabel('Akım (hm³)', fontsize=14)

    # ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.show()


def analyze_pseudo_weights(X, F, filename="PSEUDO_WEI_RESULTS.csv"):
    """
    Analyzes the Pareto front using pseudo weights, visualizes the results, and calculates
    ENG_TOT, IRG_TOT, AVE_REG_RATIO for each scenario.
    """
    # Define weight sets with labels and colors
    weight_sets = [
        (np.array([[0.33, 0.33, 0.33]]), "Dengeli Tahsis", "purple"), #"Balanced Allocation"
        (np.array([[0.8, 0.1, 0.1]]), "Enerji Üretimi Öncelikli", "orange"), #Maximizing Energy
        (np.array([[0.1, 0.8, 0.1]]), "Sürdürülebilir Sulama", "#008080"),#Sustainable Irrigation
        (np.array([[0.1, 0.1, 0.8]]), "Ekosistem Odaklı Yaklaşım", "#00BFFF"),#Eco-Centric Approach
    ]

    pseudo_weights_list, best_solutions, best_results = [], [], []

    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]

    for weights, label, color in weight_sets:
        I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
        best_solutions.append(F[I])
        pseudo_weights_list.append(pseudo_weights)

        # Calculate ENG_TOT, IRG_TOT, AVE_REG_RATIO for the best solution
        DEC_VARS = np.array([float(val) for val in X[I]]).reshape(13, 12)
        RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
        (
            ENR_OBJ,
            IRG_OBJ,
            ECO_OBJ,
            ENG_TOT,
            IRG_TOT,
            TOT_ECO_DEV,
            AVE_REG_RATIO,
            SPILLWAY,
            GEN_WATER,
            IRG_WATER,
            GEN_EN_GWH,
            POWER_MW,
            POW_MW_AVE,
            STORAGES,
            RES_EVA,
            RES_ELEV,
            OUTFLOW,
            PEN_MIN_SQTOT,
            PEN_MAX_SQTOT,
            PEN_END_SQTOT,
            PEN_ALL_TOTAL,
            STO_DIF,
            IRG_DEF_SQTOT,
            PEN_SCALE,
        ) = RESULT
        best_results.append((ENG_TOT, IRG_TOT, AVE_REG_RATIO))

    # Normalization for Visualization
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    normalized_solutions = [(sol - approx_ideal) / (approx_nadir - approx_ideal) for sol in best_solutions]

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(nF[:, 0], nF[:, 1], nF[:, 2], alpha=0.1, facecolor="grey", edgecolor="grey")
    # cmap = plt.cm.get_cmap("gray", 5)
    cmap = plt.cm.get_cmap("plasma")
    # Normalize the data to the range [0, 1]
    nF_normalized = (nF - nF.min()) / (nF.max() - nF.min())

    # Generate colors from the colormap based on the normalized data
    colors = cmap(nF_normalized[:, 1])  # Use the first column for coloring

    scatter = ax.scatter(nF[:, 0], nF[:, 1], nF[:, 2], marker="o", facecolor=colors, s=20, alpha=0.1, zorder=1)

    # Scatter plot with labels and colors for each scenario (using normalized solutions)
    for i, (solution, (weights, label, color), pseudo_weights) in enumerate(zip(normalized_solutions, weight_sets, pseudo_weights_list)):
        ax.scatter(solution[0], solution[1], solution[2], alpha=0.8, c=color, marker="o", s=50, label=label, zorder=5)

        ax.plot([solution[0], solution[0]], [solution[1], solution[1]], [0, solution[2]], c=color, linestyle="dotted", alpha=0.8)
        ax.plot([solution[0], solution[0]], [0, solution[1]], [solution[2], solution[2]], c=color, linestyle="dotted", alpha=0.8)
        ax.plot([0, solution[0]], [solution[1], solution[1]], [solution[2], solution[2]], c=color, linestyle="dotted", alpha=0.8)

    ax.set_xlabel("Enerji")
    ax.set_ylabel("Sulama")
    ax.set_zlabel("Ekosistem")

    # Set axis ticks with 0.1 interval
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_zticks(np.arange(0, 1.1, 0.1))

    # --- Virgül ile ondalık gösterim ---
    formatter = FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ','))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    ax.quiver([0.9], [1], [0], [-0.1], [0], [0], color='grey', arrow_length_ratio=0.4, length=1.2, alpha=0.7)  # x-axis
    ax.quiver([1], [0], [0.9], [0], [0], [-0.1], color='grey', arrow_length_ratio=0.4, length=1.2, alpha=0.7)  # z-axis
    ax.quiver([1], [0.9], [0], [0], [-0.1], [0], color='grey', arrow_length_ratio=0.4, length=1.2, alpha=0.7)  # y-axis

    ax.view_init(elev=30, azim=45)
    # ax.legend(ncol=len(weight_sets))
    plt.show()

    for i, (solution, (weights, label, _), pseudo_weights) in enumerate(zip(best_solutions, weight_sets, pseudo_weights_list), 1):
        print(f"Best solution {i} ({label}) regarding weights {weights}: Point {solution} ")

    for i, (weights, label, _) in enumerate(weight_sets):
        results = best_results[i]
        print(f"Best results for scenario {i + 1} ({label}) with weights {weights}:")
        print(f"  ENG_TOT: {results[0]:.4f}, IRG_TOT: {results[1]:.4f}, AVE_REG_RATIO: {results[2]:.4f}")

    # Export results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "ENG_TOT", "IRG_TOT", "AVE_REG_RATIO"])
        for (weights, label, _), results in zip(weight_sets, best_results):
            writer.writerow([label, f"{results[0]:.4f}", f"{results[1]:.4f}", f"{results[2]:.4f}"])

    # Open the exported CSV file
    os.startfile(filename)  # This will open the file with the default associated program

    return best_results  # Return the best_results list


def analyze_pseudo_weights_animated(X, F, filename="PSEUDO_WEI_RESULTS.csv", output_gif_name="pseudo_weights_rotation.gif", num_frames=90, rotation_speed=2):

    """
    Analyzes the Pareto front using pseudo weights, visualizes the results, and calculates
    ENG_TOT, IRG_TOT, AVE_REG_RATIO for each scenario.
    Generates an animated GIF of the 3D plot with rotation.

    Args:
        X (numpy array): Decision variables data.
        F (numpy array): Objective function values (Pareto front data).
        filename (str): Name of the output CSV file.
        output_gif_name (str): Name of the output GIF file.
        num_frames (int): Number of frames in the GIF.
        rotation_speed (int): Speed of rotation for the GIF.
    """
    # Define weight sets with labels and colors

    plt.rcParams["font.size"] = 10  # Adjust this value as needed

    weight_sets = [
        (np.array([[0.33, 0.33, 0.33]]), "Balanced Allocation", "purple"),
        (np.array([[0.8, 0.1, 0.1]]), "Maximizing Energy", "orange"),
        (np.array([[0.1, 0.8, 0.1]]), "Sustainable Irrigation", "#008080"),
        (np.array([[0.1, 0.1, 0.8]]), "Eco-Centric Approach", "#00BFFF"),
    ]

    pseudo_weights_list, best_solutions, best_results = [], [], []

    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]

    for weights, label, color in weight_sets:
        I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
        best_solutions.append(F[I])  # F[I] is the solution in objective space
        pseudo_weights_list.append(pseudo_weights)

        # Calculate ENG_TOT, IRG_TOT, AVE_REG_RATIO for the best solution
        # Make sure X[I] is a 1D array before reshaping, as I is an index
        dec_vars_for_sim = np.array([float(val) for val in X[I]]).reshape(13, 12) if X[I].ndim == 0 else X[I].reshape(13, 12)
        RESULT = simulation_model(dec_vars_for_sim, K_1, K_2, K_3, K_SC)
        (
            ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO,
            SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, POW_MW_AVE, STORAGES,
            RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT,
            PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE
        ) = RESULT
        best_results.append((ENG_TOT, IRG_TOT, AVE_REG_RATIO))

    # Normalization for Visualization (Applies to all F and best_solutions)
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    # Handle cases where min_val == max_val to prevent division by zero
    range_vals = approx_nadir - approx_ideal
    range_vals[range_vals == 0] = 1

    nF = (F - approx_ideal) / range_vals  # Normalized full Pareto Front
    normalized_solutions = [(sol - approx_ideal) / range_vals for sol in best_solutions]  # Normalized best solutions

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Define colormap for background scatter
    cmap = plt.cm.get_cmap("plasma")

    # Normalize data for coloring the background scatter
    # Use one of the objectives (e.g., the second objective F[:,1] for coloring)
    nF_color_normalized = (nF[:, 1] - nF[:, 1].min()) / (nF[:, 1].max() - nF[:, 1].min())
    colors = cmap(nF_color_normalized)

    # Plot the full normalized Pareto front (background scatter)
    scatter = ax.scatter(nF[:, 0], nF[:, 1], nF[:, 2], marker="o", facecolor=colors, s=10, alpha=0.1, zorder=1)

    # Scatter plot with labels and colors for each scenario (using normalized solutions)
    for i, (solution, (weights, label, color), pseudo_weights) in enumerate(zip(normalized_solutions, weight_sets, pseudo_weights_list)):
        ax.scatter(solution[0], solution[1], solution[2], alpha=0.8, c=color, marker="o", s=20, label=label, zorder=5)

        # Plot projection lines to axes for better 3D visualization
        # ax.plot([solution[0], solution[0]], [solution[1], solution[1]], [0, solution[2]], c=color, linestyle="dotted", alpha=0.8)
        # ax.plot([solution[0], solution[0]], [0, solution[1]], [solution[2], solution[2]], c=color, linestyle="dotted", alpha=0.8)
        # ax.plot([0, solution[0]], [solution[1], solution[1]], [solution[2], solution[2]], c=color, linestyle="dotted", alpha=0.8)

    # Set axis labels with padding
    ax.set_xlabel("Enerji Üretim Açığı", labelpad=5) #Normalized Energy
    ax.set_ylabel("Sulama Açığı", labelpad=5) #Normalized Irrigation
    ax.set_zlabel("Doğal Akımdan Sapma", labelpad=5) #Normalized Ecology

    # Set axis ticks with 0.1 interval for the normalized [0,1] range
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_zticks(np.arange(0, 1.1, 0.2))

    # Set explicit limits for normalized axes with a small buffer
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)

    # Add quiver arrows for axis direction (assuming minimization towards origin)
    # The arrows indicate the direction of improvement for normalized objectives.
    # From [almost 1] to [0]
    # ax.quiver(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, color='grey', arrow_length_ratio=0.08, length=1.0, alpha=0.7)  # X (Energy)
    # ax.quiver(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, color='grey', arrow_length_ratio=0.08, length=1.0, alpha=0.7)  # Y (Irrigation)
    # ax.quiver(0.0, 0.0, 1.0, 0.0, 0.0, -1.0, color='grey', arrow_length_ratio=0.08, length=1.0, alpha=0.7)  # Z (Ecology)

    ax.view_init(elev=20, azim=45)  # Initial view angle
    # ax.legend(ncol=len(weight_sets), loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=False)  # Adjust legend position
    plt.tight_layout()  # Adjust plot parameters for a tight layout

    # --- GIF ANIMATION PART ---
    filenames = []
    for i in range(num_frames):
        ax.view_init(elev=30, azim=i * rotation_speed)  # Rotate azimuth
        frame_filename = f'frame_{i:03d}.png'
        plt.savefig(frame_filename, dpi=150)  # Save frame
        filenames.append(frame_filename)

    with imageio.get_writer(output_gif_name, mode='I', duration=0.1) as writer:
        for frame_filename in filenames:
            image = imageio.imread(frame_filename)
            writer.append_data(image)

    for frame_filename in set(filenames):  # Clean up temporary files
        os.remove(frame_filename)

    plt.close(fig)  # Close the figure to free memory
    print(f"GIF '{output_gif_name}' created successfully!")
    # --- END GIF ANIMATION PART ---

    # Print best results to console
    for i, (solution, (weights, label, _), pseudo_weights) in enumerate(zip(best_solutions, weight_sets, pseudo_weights_list), 1):
        print(f"Best solution {i} ({label}) regarding weights {weights}: Point {solution} ")

    for i, (weights, label, _) in enumerate(weight_sets):
        results = best_results[i]
        print(f"Best results for scenario {i + 1} ({label}) with weights {weights}:")
        print(f"  ENG_TOT: {results[0]:.4f}, IRG_TOT: {results[1]:.4f}, AVE_REG_RATIO: {results[2]:.4f}")

    # Export results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "ENG_TOT", "IRG_TOT", "AVE_REG_RATIO"])
        for (weights, label, _), results in zip(weight_sets, best_results):
            writer.writerow([label, f"{results[0]:.4f}", f"{results[1]:.4f}", f"{results[2]:.4f}"])

    # Open the exported CSV file
    # os.startfile(filename) # Uncomment if you want to automatically open the CSV

    return best_results


def analyze_pseudo_weights_coordinate_plot(X, F, filename="PSEUDO_WEI_RESULTS.csv"):
    """
    Analyzes the Pareto front using pseudo weights, visualizes the results using a parallel coordinates plot,
    and calculates ENG_TOT, IRG_TOT, AVE_REG_RATIO for each scenario.

    Args:
        X (numpy array): Decision variables data.
        F (numpy array): Objective function values (Pareto front data).
        filename (str): Name of the output CSV file.
    """
    plt.rcParams["font.size"] = 10  # Adjust global font size

    # Define weight sets with labels and colors
    weight_sets = [
        (np.array([[0.33, 0.33, 0.33]]), "Balanced Allocation", "purple"),
        (np.array([[0.8, 0.1, 0.1]]), "Maximizing Energy", "orange"),
        (np.array([[0.1, 0.8, 0.1]]), "Sustainable Irrigation", "#008080"),  # Teal
        (np.array([[0.1, 0.1, 0.8]]), "Eco-Centric Approach", "#00BFFF"),  # Deep Sky Blue
    ]

    pseudo_weights_list, best_solutions, best_results = [], [], []

    total_iter, iter_name, coef_values = read_penalty_coef("EXP_NSGA3_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]

    for weights, label, color in weight_sets:
        I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
        best_solutions.append(F[I])
        pseudo_weights_list.append(pseudo_weights)

        dec_vars_for_sim = np.array([float(val) for val in X[I]]).reshape(13, 12) if X[I].ndim == 0 else X[I].reshape(13, 12)
        RESULT = simulation_model(dec_vars_for_sim, K_1, K_2, K_3, K_SC)
        (
            ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO,
            SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, POW_MW_AVE, STORAGES,
            RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT,
            PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE
        ) = RESULT
        best_results.append((ENG_TOT, IRG_TOT, AVE_REG_RATIO))

    # --- Normalization for Visualization ---
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    range_vals = approx_nadir - approx_ideal
    range_vals[range_vals == 0] = 1

    nF = (F - approx_ideal) / range_vals  # Normalized full Pareto Front
    normalized_solutions = [(sol - approx_ideal) / range_vals for sol in best_solutions]
    # --- End Normalization ---

    # --- Parallel Coordinates Plotting ---
    # Define feature names (objectives) for parallel coordinates axes
    # Assuming F's columns are in order: [Energy, Irrigation, Ecology]
    ynames = ["Enerji", "Sulama", "Ekosistem"] #["Energy", "Irrigation", "Ecology"]
    num_features = nF.shape[1]

    fig, host = plt.subplots(figsize=(10, 5))
    axes = [host] + [host.twinx() for _ in range(num_features - 1)]

    # Set common y-axis limits for normalized data and configure axes spines
    for i, ax in enumerate(axes):
        ax.set_ylim(-0.05, 1.05)  # Add small buffer for normalized range
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (num_features - 1)))

    host.set_xlim(0, num_features - 1)
    host.set_xticks(range(num_features))
    host.set_xticklabels(ynames, fontsize=12, rotation=0, ha='right')
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)

    # Plot the full normalized Pareto front (background lines)
    for k in range(nF.shape[0]):
        verts = list(zip(range(num_features), nF[k, :]))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor='none', lw=1, alpha=0.1, edgecolor='lightgrey'
        )
        host.add_patch(patch)

    # Plot the specific best solutions from pseudo-weights
    for i, (solution, (weights, label, color), pseudo_weights) in enumerate(zip(normalized_solutions, weight_sets, pseudo_weights_list)):
        verts = list(zip(range(num_features), solution))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor='none', lw=1.5, alpha=0.9, edgecolor=color,
            label=label
        )
        host.add_patch(patch)
        host.plot(num_features - 1, solution[-1], marker='o', markersize=5, color=color, zorder=10)

    # Create legend
    handles, labels = host.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(weight_sets), fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # --- Eksenlerde ondalık ayırıcıyı virgül yap ---
    formatter = FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ','))
    for ax in axes:
        ax.yaxis.set_major_formatter(formatter)

    plt.show()
    # --- End Parallel Coordinates Plotting ---

    # Print best results to console
    for i, (solution, (weights, label, _), pseudo_weights) in enumerate(zip(best_solutions, weight_sets, pseudo_weights_list), 1):
        print(f"Best solution {i} ({label}) regarding weights {weights}: Point {solution} ")

    for i, (weights, label, _) in enumerate(weight_sets):
        results = best_results[i]
        print(f"Best results for scenario {i + 1} ({label}) with weights {weights}:")
        print(f"  ENG_TOT: {results[0]:.4f}, IRG_TOT: {results[1]:.4f}, AVE_REG_RATIO: {results[2]:.4f}")

    # Export results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "ENG_TOT", "IRG_TOT", "AVE_REG_RATIO"])
        for (weights, label, _), results in zip(weight_sets, best_results):
            writer.writerow([label, f"{results[0]:.4f}", f"{results[1]:.4f}", f"{results[2]:.4f}"])

    return best_results

def get_results_from_pseudo_weights(X, F, filename="PSEUDO_WEI_OBJ.csv"):
    # Multi-Criteria Decision Making - Pseudo
    # MERGE FOR SUBBASIN AND EXPORT VALUE
    # Define weight sets with labels and colors
    weight_sets = [
        (np.array([[0.33, 0.33, 0.33]]), "Balanced", "purple"),
        (np.array([[0.8, 0.1, 0.1]]), "Maximizing Energy", "orange"),
        (np.array([[0.1, 0.8, 0.1]]), "Sustainable Irrigation", "#008080"),
        (np.array([[0.1, 0.1, 0.8]]), "Eco-Centric Approach", "#00BFFF"),
    ]

    pseudo_weights_list, best_solutions, best_results = [], [], []

    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GROUP NAME", "RES_EVA", "OUTFLOW", "GEN_EN_GWH", "IRG_WATER", "Weight Set"])

        for weights, label, color in weight_sets:
            I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
            best_solutions.append(F[I])
            pseudo_weights_list.append(pseudo_weights)

            DEC_VARS = np.array([float(val) for val in X[I]]).reshape(13, 12)
            RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
            (
                ENR_OBJ,
                IRG_OBJ,
                ECO_OBJ,
                ENG_TOT,
                IRG_TOT,
                TOT_ECO_DEV,
                AVE_REG_RATIO,
                SPILLWAY,
                GEN_WATER,
                IRG_WATER,
                GEN_EN_GWH,
                POWER_MW,
                POW_MW_AVE,
                STORAGES,
                RES_EVA,
                RES_ELEV,
                OUTFLOW,
                PEN_MIN_SQTOT,
                PEN_MAX_SQTOT,
                PEN_END_SQTOT,
                PEN_ALL_TOTAL,
                STO_DIF,
                IRG_DEF_SQTOT,
                PEN_SCALE,
            ) = RESULT

            basin_names = get_data_string("CH_NAMES")[:-1]
            group_names = [
                "Upper Sakarya", "Porsuk", "Ankara", "Middle Sakarya", "Middle Sakarya",
                "Kirmir", "Middle Sakarya", "Middle Sakarya", "Middle Sakarya",
                "Middle Sakarya", "Göksu", "Middle Sakarya", "Lower Sakarya"
            ]
            unique_group_names = list(set(group_names))
            # Sort the unique group names as requested
            group_order = ["Upper Sakarya", "Porsuk", "Ankara", "Kirmir", "Middle Sakarya", "Göksu", "Lower Sakarya"]
            unique_group_names = sorted(unique_group_names, key=lambda x: group_order.index(x))

            group_data = {}
            for i, group in enumerate(group_names):
                if group not in group_data:
                    group_data[group] = {
                        "RES_EVA": 0,
                        "OUTFLOW": 0,  # Initialize OUTFLOW to 0 for summing
                        "GEN_EN_GWH": 0,
                        "IRG_WATER": 0
                    }
                group_data[group]["RES_EVA"] += np.sum(RES_EVA[i, :])

                if group == "Middle Sakarya" and basin_names[i] == "Pamukova":
                    group_data[group]["OUTFLOW"] = np.sum(OUTFLOW[i, :])  # Take OUTFLOW from "Pamukova"
                elif group != "Middle Sakarya":
                    group_data[group]["OUTFLOW"] += np.sum(OUTFLOW[i, :])  # Sum OUTFLOW for other groups

                group_data[group]["GEN_EN_GWH"] += np.sum(GEN_EN_GWH[i, :])
                group_data[group]["IRG_WATER"] += np.sum(IRG_WATER[i, :])

            for group in unique_group_names:
                writer.writerow([group,
                                 f"{group_data[group]['RES_EVA']:.4f}",
                                 f"{group_data[group]['OUTFLOW']:.4f}",
                                 f"{group_data[group]['GEN_EN_GWH']:.4f}",
                                 f"{group_data[group]['IRG_WATER']:.4f}",
                                 label])  # Add the weight set label to the row

    # Open the exported CSV file
    os.startfile(filename)

    # Return the desired 2D arrays
    return RES_EVA, OUTFLOW, GEN_EN_GWH, IRG_WATER

def get_balanced_irrigation(decision_variables, objective_function_values, balanced_weights=None):
    """
    Get the irrigation water results for the best solution based on balanced weights.

    Parameters:
    - decision_variables: Array of decision variables.
    - objective_function_values: Array of objective function values.
    - balanced_weights: Optional array of weights for the objectives (default is [0.33, 0.33, 0.33]).

    Returns:
    - irg_water: Balanced irrigation water results.
    """
    # Set default balanced weights if not provided
    if balanced_weights is None:
        balanced_weights = np.array([0.33, 0.33, 0.33])

    # Ensure balanced_weights is a 2D array
    balanced_weights = np.array([balanced_weights])

    # Calculate pseudo weights
    _, pseudo_weights = PseudoWeights(balanced_weights).do(objective_function_values, return_pseudo_weights=True)

    # Get the best solution index for balanced weights
    best_solution_index = PseudoWeights(balanced_weights).do(objective_function_values)

    # Extract the decision variables for the best solution
    decision_variables = np.array([float(val) for val in decision_variables[best_solution_index]]).reshape(13, 12)

    # Get simulation results
    total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    result = simulation_model(decision_variables, K_1, K_2, K_3, K_SC)

    # Extract irrigation water
    irg_water = result[9]  # Assuming IRG_WATER is at index 9
    return irg_water

def get_balanced_energy_production(decision_variables, objective_function_values):
    # Define balanced weights
    balanced_weights = np.array([[0.33, 0.33, 0.33]])

    # Calculate pseudo weights
    _, pseudo_weights = PseudoWeights(balanced_weights).do(objective_function_values, return_pseudo_weights=True)

    # Get the best solution index for balanced weights
    best_solution_index = PseudoWeights(balanced_weights).do(objective_function_values)

    # Extract the decision variables for the best solution
    decision_variables = np.array([float(val) for val in decision_variables[best_solution_index]]).reshape(13, 12)

    # Get simulation results
    total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    result = simulation_model(decision_variables, K_1, K_2, K_3, K_SC)

    # Extract generated energy and observed firm energy
    generated_energy_gwh = result[10]  # Assuming GEN_EN_GWH is at index 10
    return generated_energy_gwh

def get_maxenrgy_energy_production(decision_variables, objective_function_values):
    # Define Maximizing Energy
    balanced_weights = np.array([[0.8, 0.1, 0.1]])

    # Calculate pseudo weights
    _, pseudo_weights = PseudoWeights(balanced_weights).do(objective_function_values, return_pseudo_weights=True)

    # Get the best solution index for balanced weights
    best_solution_index = PseudoWeights(balanced_weights).do(objective_function_values)

    # Extract the decision variables for the best solution
    decision_variables = np.array([float(val) for val in decision_variables[best_solution_index]]).reshape(13, 12)

    # Get simulation results
    total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    result = simulation_model(decision_variables, K_1, K_2, K_3, K_SC)

    # Extract generated energy and observed firm energy
    generated_energy_gwh = result[10]  # Assuming GEN_EN_GWH is at index 10
    return generated_energy_gwh

def get_balanced_ecology(decision_variables, objective_function_values):
    # Define balanced weights
    balanced_weights = np.array([[0.33, 0.33, 0.33]])

    # Calculate pseudo weights
    _, pseudo_weights = PseudoWeights(balanced_weights).do(objective_function_values, return_pseudo_weights=True)

    # Get the best solution index for balanced weights
    best_solution_index = PseudoWeights(balanced_weights).do(objective_function_values)

    # Extract the decision variables for the best solution
    decision_variables = np.array([float(val) for val in decision_variables[best_solution_index]]).reshape(13, 12)

    # Get simulation results
    total_iter, iter_name, coef_values = read_penalty_coef("SIM_NOR_NSG2_PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    result = simulation_model(decision_variables, K_1, K_2, K_3, K_SC)
    flow = result[16]
    return flow


from pymoo.mcdm.compromise_programming import CompromiseProgramming
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.decomposition.asf import ASF


def analyze_pseudo_weights_worst_best(X, F, filename="PSEUDO_WEI_RESULTS.csv"):
    """
    Analyzes the Pareto front using pseudo weights to find the best and worst cases,
    visualizes the results, and calculates ENG_TOT, IRG_TOT, AVE_REG_RATIO for each scenario.
    Also exports detailed results to a CSV file.
    """
    # Define weight sets with labels and colors
    weight_sets = [
        (np.array([[1.0, 0.0, 0.0]]), "Best Case (Energy)", "red"),  # Prioritize only Energy
        (np.array([[0.0, 1.0, 0.0]]), "Best Case (Irrigation)", "green"),  # Prioritize only Irrigation
        (np.array([[0.0, 0.0, 1.0]]), "Best Case (Ecology)", "blue"),  # Prioritize only Ecology
        (np.array([[0.0, 0.0, 1.0]]), "Worst Case (Energy)", "red"),  # Prioritize Ecology (worst for Energy)
        (np.array([[1.0, 0.0, 0.0]]), "Worst Case (Irrigation)", "green"),  # Prioritize Energy (worst for Irrigation)
        (np.array([[0.0, 1.0, 0.0]]), "Worst Case (Ecology)", "blue"),  # Prioritize Irrigation (worst for Ecology)
    ]

    pseudo_weights_list, best_solutions, best_results = [], [], []

    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF_SIM1_")
    K_1, K_2, K_3, K_SC = coef_values[0]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GROUP NAME", "RES_EVA", "OUTFLOW", "GEN_EN_GWH", "IRG_WATER", "Weight Set"])

        for weights, label, color in weight_sets:
            I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
            best_solutions.append(F[I])
            pseudo_weights_list.append(pseudo_weights)

            DEC_VARS = np.array([float(val) for val in X[I]]).reshape(13, 12)
            RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
            (
                ENR_OBJ,
                IRG_OBJ,
                ECO_OBJ,
                ENG_TOT,
                IRG_TOT,
                TOT_ECO_DEV,
                AVE_REG_RATIO,
                SPILLWAY,
                GEN_WATER,
                IRG_WATER,
                GEN_EN_GWH,
                POWER_MW,
                POW_MW_AVE,
                STORAGES,
                RES_EVA,
                RES_ELEV,
                OUTFLOW,
                PEN_MIN_SQTOT,
                PEN_MAX_SQTOT,
                PEN_END_SQTOT,
                PEN_ALL_TOTAL,
                STO_DIF,
                IRG_DEF_SQTOT,
                PEN_SCALE,
            ) = RESULT
            best_results.append((ENG_TOT, IRG_TOT, AVE_REG_RATIO))

            basin_names = get_data_string("CH_NAMES")[:-1]
            group_names = [
                "Upper Sakarya", "Porsuk", "Ankara", "Middle Sakarya", "Middle Sakarya",
                "Kirmir", "Middle Sakarya", "Middle Sakarya", "Middle Sakarya",
                "Middle Sakarya", "Göksu", "Middle Sakarya", "Lower Sakarya"
            ]
            unique_group_names = list(set(group_names))
            # Sort the unique group names as requested
            group_order = ["Upper Sakarya", "Porsuk", "Ankara", "Kirmir", "Middle Sakarya", "Göksu", "Lower Sakarya"]
            unique_group_names = sorted(unique_group_names, key=lambda x: group_order.index(x))

            group_data = {}
            for i, group in enumerate(group_names):
                if group not in group_data:
                    group_data[group] = {
                        "RES_EVA": 0,
                        "OUTFLOW": 0,  # Initialize OUTFLOW to 0 for summing
                        "GEN_EN_GWH": 0,
                        "IRG_WATER": 0
                    }
                group_data[group]["RES_EVA"] += np.sum(RES_EVA[i, :])

                if group == "Middle Sakarya" and basin_names[i] == "Pamukova":
                    group_data[group]["OUTFLOW"] = np.sum(OUTFLOW[i, :])  # Take OUTFLOW from "Pamukova"
                elif group != "Middle Sakarya":
                    group_data[group]["OUTFLOW"] += np.sum(OUTFLOW[i, :])  # Sum OUTFLOW for other groups

                group_data[group]["GEN_EN_GWH"] += np.sum(GEN_EN_GWH[i, :])
                group_data[group]["IRG_WATER"] += np.sum(IRG_WATER[i, :])

            for group in unique_group_names:
                writer.writerow([group,
                                 f"{group_data[group]['RES_EVA']:.4f}",
                                 f"{group_data[group]['OUTFLOW']:.4f}",
                                 f"{group_data[group]['GEN_EN_GWH']:.4f}",
                                 f"{group_data[group]['IRG_WATER']:.4f}",
                                 label])  # Add the weight set label to the row

    # Open the exported CSV file
    os.startfile(filename)

    # --- Normalization for Visualization ---
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)  # Normalize the objective values
    normalized_solutions = [(sol - approx_ideal) / (approx_nadir - approx_ideal) for sol in best_solutions]
    # --- End of Normalization ---

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with grey points
    ax.scatter(nF[:, 0], nF[:, 1], nF[:, 2], marker="o", facecolor="grey", edgecolor="grey", s=20, alpha=0.1, zorder=1)

    # Scatter plot with labels and colors for each scenario (using normalized solutions)
    for i, (solution, (weights, label, color), pseudo_weights) in enumerate(zip(normalized_solutions, weight_sets, pseudo_weights_list)):
        ax.scatter(solution[0], solution[1], solution[2], alpha=1, c=color, marker="o", s=20, label=label, zorder=2)

    ax.set_xlabel("Energy (Normalized)")
    ax.set_ylabel("Irrigation (Normalized)")
    ax.set_zlabel("Ecology (Normalized)")
    ax.legend()
    plt.show()

    for i, (solution, (weights, label, _), pseudo_weights) in enumerate(zip(best_solutions, weight_sets, pseudo_weights_list), 1):
        print(f"Best solution {i} ({label}) regarding weights {weights}: Point {solution} ")

    for i, (weights, label, _) in enumerate(weight_sets):
        results = best_results[i]
        print(f"Best results for scenario {i + 1} ({label}) with weights {weights}:")
        print(f"  ENG_TOT: {results[0]:.4f}, IRG_TOT: {results[1]:.4f}, AVE_REG_RATIO: {results[2]:.4f}")

    # Export results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "ENG_TOT", "IRG_TOT", "AVE_REG_RATIO"])
        for (weights, label, _), results in zip(weight_sets, best_results):
            writer.writerow([label, f"{results[0]:.4f}", f"{results[1]:.4f}", f"{results[2]:.4f}"])

    # Open the exported CSV file
    os.startfile(filename)  # This will open the file with the default associated program
    # filename = "EXP_PSEWEI_SCE_NSG3.csv"

    return best_results  # Return the best_results list


def get_min_max_objectives(F):
    # Get the minimum values for each objective
    min_values = np.min(F, axis=0)
    # Get the maximum values for each objective
    max_values = np.max(F, axis=0)
    print("Minimum values:", min_values)
    print("Maximum values:", max_values)
    return min_values, max_values


def get_results_from_compromise(X, F, filename="EXP_COMP_PROG_RESULTS.csv"):
    # Multi-Criteria Decision Making - Compromise Programming
    weights = np.array([0.33, 0.33, 0.34])
    decomp = ASF()

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    # Find the best solution
    i = decomp.do(nF, 1 / weights).argmin()  # Pass ideal_point as keyword argument

    # Best overall solution
    best_solution = F[i]

    # Calculate ENG_TOT, IRG_TOT, AVE_REG_RATIO for the best solution
    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    DEC_VARS = np.array([float(val) for val in X[i]]).reshape(13, 12)
    RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
    (
        ENR_OBJ,
        IRG_OBJ,
        ECO_OBJ,
        ENG_TOT,
        IRG_TOT,
        TOT_ECO_DEV,
        AVE_REG_RATIO,
        SPILLWAY,
        GEN_WATER,
        IRG_WATER,
        GEN_EN_GWH,
        POWER_MW,
        POW_MW_AVE,
        STORAGES,
        RES_EVA,
        RES_ELEV,
        OUTFLOW,
        PEN_MIN_SQTOT,
        PEN_MAX_SQTOT,
        PEN_END_SQTOT,
        PEN_ALL_TOTAL,
        STO_DIF,
        IRG_DEF_SQTOT,
        PEN_SCALE,
    ) = RESULT

    basin_names = get_data_string("CH_NAMES")[:-1]
    group_names = [
        "Upper Sakarya", "Porsuk", "Ankara", "Middle Sakarya", "Middle Sakarya",
        "Kirmir", "Middle Sakarya", "Middle Sakarya", "Middle Sakarya",
        "Middle Sakarya", "Göksu", "Middle Sakarya", "Lower Sakarya"
    ]
    unique_group_names = list(set(group_names))
    # Sort the unique group names as requested
    group_order = ["Upper Sakarya", "Porsuk", "Ankara", "Kirmir", "Middle Sakarya", "Göksu", "Lower Sakarya"]
    unique_group_names = sorted(unique_group_names, key=lambda x: group_order.index(x))

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GROUP NAME", "RES_EVA", "OUTFLOW", "GEN_EN_GWH", "IRG_WATER"])

        group_data = {}
        for i, group in enumerate(group_names):
            if group not in group_data:
                group_data[group] = {
                    "RES_EVA": 0,
                    "OUTFLOW": 0,  # Initialize OUTFLOW to 0 for summing
                    "GEN_EN_GWH": 0,
                    "IRG_WATER": 0
                }
            group_data[group]["RES_EVA"] += np.sum(RES_EVA[i, :])

            if group == "Middle Sakarya" and basin_names[i] == "Pamukova":
                group_data[group]["OUTFLOW"] = np.sum(OUTFLOW[i, :])  # Take OUTFLOW from "Pamukova"
            elif group != "Middle Sakarya":
                group_data[group]["OUTFLOW"] += np.sum(OUTFLOW[i, :])  # Sum OUTFLOW for other groups

            group_data[group]["GEN_EN_GWH"] += np.sum(GEN_EN_GWH[i, :])
            group_data[group]["IRG_WATER"] += np.sum(IRG_WATER[i, :])

        for group in unique_group_names:
            writer.writerow([group,
                             f"{group_data[group]['RES_EVA']:.4f}",
                             f"{group_data[group]['OUTFLOW']:.4f}",
                             f"{group_data[group]['GEN_EN_GWH']:.4f}",
                             f"{group_data[group]['IRG_WATER']:.4f}"])

    # Open the exported CSV file
    os.startfile(filename)

    # Return the desired 2D arrays
    return RES_EVA, OUTFLOW, GEN_EN_GWH, IRG_WATER


def BCKplot_3d_pareto_compare(RES_F1, RES_F2, color_1, color_2):
    """
    Plots a 3D scatter plot of two Pareto fronts.

    Args:
        RES_F1 (numpy array): Pareto front data for the first simulation.
        RES_F2 (numpy array): Pareto front data for the second simulation.
    """

    fig = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.scatter(
        RES_F1[:, 2],
        RES_F1[:, 1],
        RES_F1[:, 0],
        alpha=0.5,
        c=color_1,
        marker="o",
        s=10,
        label="NSGA-II",
        edgecolors="none",
    )
    ax3d.scatter(
        RES_F2[:, 2],
        RES_F2[:, 1],
        RES_F2[:, 0],
        alpha=0.5,
        c=color_2,
        marker="o",
        s=10,
        label="NSGA-III",
        edgecolors="none",
    )

    ax3d.set_xlabel("Dev. from Natural Flow") #Deviation from Natural Flow
    ax3d.set_ylabel("Irrigation Def.") #Irrigation Deficits
    ax3d.set_zlabel("Pot. Energy Def.") #Deficit from the Potential Energy
    # ax3d.legend()
    # plt.title("3D Pareto Front Comparison")
    plt.show()
def plot_3d_pareto_compare(RES_F1, RES_F2, color_1_hues, color_2_hues):
    """
    Plots a 3D scatter plot of two Pareto fronts, with each front divided into 3 classes,
    and allows user to define the color hues.

    Args:
        RES_F1 (numpy array): Pareto front data for the first simulation.
        RES_F2 (numpy array): Pareto front data for the second simulation.
        color_1_hues (list): List of 3 colors for the first simulation classes.
        color_2_hues (list): List of 3 colors for the second simulation classes.
    """

    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # Function to divide data into 3 classes based on a chosen axis
    def classify_data(data, axis_index):
        sorted_indices = np.argsort(data[:, axis_index])
        classes = np.array_split(sorted_indices, 3)
        return classes

    # Classify data for each simulation
    classes_1 = classify_data(RES_F1, 2)  # Classify based on the first objective (Pot. Energy Def.)
    classes_2 = classify_data(RES_F2, 2)  # Classify based on the first objective (Pot. Energy Def.)

    # Plot each class with user-defined color hues
    for i, class_indices in enumerate(classes_1):
        ax3d.scatter(
            RES_F1[class_indices, 2],
            RES_F1[class_indices, 1],
            RES_F1[class_indices, 0],
            alpha=0.8,
            c=color_1_hues[i],
            marker="o",
            s=3,
            edgecolors="none",
            label=f"NSGA-II {i+1}" if i == 0 else "",
        )

    for i, class_indices in enumerate(classes_2):
        ax3d.scatter(
            RES_F2[class_indices, 2],
            RES_F2[class_indices, 1],
            RES_F2[class_indices, 0],
            alpha=0.8,
            c=color_2_hues[i],
            marker="o",
            s=3,
            edgecolors="none",
            label=f"NSGA-III {i+1}" if i == 0 else "",
        )

    ax3d.set_xlabel("Dev. from Natural Flow")
    ax3d.set_ylabel("Irrigation Def.")
    ax3d.set_zlabel("Pot. Energy Def.")
    ax3d.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_3d_pareto_compare_normalize(RES_F1, RES_F2, color_1_hues, color_2_hues):
    """
    Plots a 3D scatter plot of two Pareto fronts, with each front divided into 3 classes,
    and allows the user to define the color hues.

    The data is normalized using Min-Max scaling to a unified [0, 1] range before plotting
    to ensure meaningful visual comparison.

    Args:
        RES_F1 (numpy array): Pareto front data for the first simulation (e.g., NSGA-II).
        RES_F2 (numpy array): Pareto front data for the second simulation (e.g., NSGA-III).
        color_1_hues (list): List of 3 colors for the first simulation classes.
        color_2_hues (list): List of 3 colors for the second simulation classes.
    """

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # --- Normalization Step (Copied from the animated function) ---

    # 1. Combine data from both fronts to find global min/max for each objective
    combined_data = np.vstack((RES_F1, RES_F2))

    # 2. Calculate min and max for each objective across all combined data
    min_vals = np.min(combined_data, axis=0)
    max_vals = np.max(combined_data, axis=0)

    # 3. Handle zero range (avoid division by zero)
    range_vals = max_vals - min_vals
    # Replace 0 range with 1 to avoid /0. This effectively prevents scaling if data is constant.
    range_vals[range_vals == 0] = 1

    # 4. Apply Min-Max Normalization to both Pareto fronts
    # Normalized_X = (X - min_X) / (max_X - min_X)
    RES_F1_normalized = (RES_F1 - min_vals) / range_vals
    RES_F2_normalized = (RES_F2 - min_vals) / range_vals

    # --- End Normalization Step ---

    # Function to divide data into 3 classes based on a chosen axis (now uses normalized data)
    def classify_data(data, axis_index):
        # We classify based on the normalized data to ensure balanced classes across the full range
        sorted_indices = np.argsort(data[:, axis_index])
        classes = np.array_split(sorted_indices, 3)
        return classes

    # Assuming objectives are ordered (0, 1, 2) in the input arrays:
    # 0: Dev. from Natural Flow
    # 1: Irrigation Def.
    # 2: Pot. Energy Def.

    # Classification is done based on the *first objective in the original data order* (index 2)
    classes_1 = classify_data(RES_F1_normalized, 2)
    classes_2 = classify_data(RES_F2_normalized, 2)

    # --- Plotting uses the Normalized Data ---

    # Plot NSGA-II (RES_F1)
    for i, class_indices in enumerate(classes_1):
        ax3d.scatter(
            RES_F1_normalized[class_indices, 2],  # X-axis (Pot. Energy Def.)
            RES_F1_normalized[class_indices, 1],  # Y-axis (Irrigation Def.)
            RES_F1_normalized[class_indices, 0],  # Z-axis (Dev. from Natural Flow)
            alpha=0.8,
            c=color_1_hues[i],
            marker="o",
            s=15,
            edgecolors="none",
            label=f"NSGA-II Class {i + 1}" if i == 0 else "",
        )

    # Plot NSGA-III (RES_F2)
    for i, class_indices in enumerate(classes_2):
        ax3d.scatter(
            RES_F2_normalized[class_indices, 2],  # X-axis (Pot. Energy Def.)
            RES_F2_normalized[class_indices, 1],  # Y-axis (Irrigation Def.)
            RES_F2_normalized[class_indices, 0],  # Z-axis (Dev. from Natural Flow)
            alpha=0.8,
            c=color_2_hues[i],
            marker="o",
            s=15,
            edgecolors="none",
            label=f"NSGA-III Class {i + 1}" if i == 0 else "",
        )

    # Labels for the axes (based on the plotting order [2, 1, 0] from the original function)
    # ax3d.set_xlabel("Normalized Pot. Energy Def.")
    # ax3d.set_ylabel("Normalized Irrigation Def.")
    # ax3d.set_zlabel("Normalized Dev. from Natural Flow")

    # Set explicit limits for normalized axes
    ax3d.set_xlim(-0.05, 1.05)
    ax3d.set_ylim(-0.05, 1.05)
    ax3d.set_zlim(-0.05, 1.05)

    # --- Format axes with comma as decimal separator ---
    formatter = FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ',')) #nokta virgul decimal
    ax3d.xaxis.set_major_formatter(formatter)
    ax3d.yaxis.set_major_formatter(formatter)
    ax3d.zaxis.set_major_formatter(formatter)

    # ax3d.legend()
    plt.show()


# Note: The original axes labels were mismatched with the indexing in the
# animated function ([2, 1, 0] vs labels "Dev. from Natural Flow", "Irrigation Def.", "Pot. Energy Def.").
# I have corrected the labels here to match the indexing used for plotting:
# X-axis is index 2, Y-axis is index 1, Z-axis is index 0.

import matplotlib.pyplot as plt
import numpy as np
import imageio # You'll need to install this library: pip install imageio


def plot_3d_pareto_compare_animated(RES_F1, RES_F2, color_1_hues, color_2_hues, output_gif_name="3d_pareto_rotation.gif", num_frames=60,
                                    rotation_speed=1):
    """
    Plots a 3D scatter plot of two Pareto fronts, with each front divided into 3 classes,
    allows user to define the color hues, and saves an animated GIF with rotation.
    Normalizes axis values using Min-Max scaling to a [0, 1] range.

    Args:
        RES_F1 (numpy array): Pareto front data for the first simulation.
        RES_F2 (numpy array): Pareto front data for the second simulation.
        color_1_hues (list): List of 3 colors for the first simulation classes.
        color_2_hues (list): List of 3 colors for the second simulation classes.
        output_gif_name (str): Name of the output GIF file.
        num_frames (int): Number of frames in the GIF. Higher means smoother animation.
        rotation_speed (int): Speed of rotation. Higher means faster rotation per frame.
    """

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # --- Normalization Step ---
    # Combine data from both fronts to find global min/max for each objective
    combined_data = np.vstack((RES_F1, RES_F2))

    # Calculate min and max for each objective across all combined data
    min_vals = np.min(combined_data, axis=0)
    max_vals = np.max(combined_data, axis=0)

    # Avoid division by zero if an objective has no range
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Replace 0 with 1 to avoid /0, effectively no scaling if constant

    # Apply Min-Max Normalization to both Pareto fronts
    # Normalized_X = (X - min_X) / (max_X - min_X)
    RES_F1_normalized = (RES_F1 - min_vals) / range_vals
    RES_F2_normalized = (RES_F2 - min_vals) / range_vals

    # --- End Normalization Step ---

    # Function to divide data into 3 classes based on a chosen axis (now uses normalized data)
    def classify_data(data, axis_index):
        sorted_indices = np.argsort(data[:, axis_index])
        classes = np.array_split(sorted_indices, 3)
        return classes

    # Classify data for each simulation (using normalized data for classification)
    # Assuming objectives are: 0: Dev. from Natural Flow, 1: Pot. Energy Def., 2: Irrigation Def.
    # Your plotting order for scatter is (index 2, index 1, index 0) which is (Pot. Energy Def., Irrigation Def., Dev. from Natural Flow)
    # So axis_index=2 still means classification based on Pot. Energy Def.
    classes_1 = classify_data(RES_F1_normalized, 2)
    classes_2 = classify_data(RES_F2_normalized, 2)

    # Plot each class with user-defined color hues (initial plot)
    for i, class_indices in enumerate(classes_1):
        ax3d.scatter(
            RES_F1_normalized[class_indices, 2],  # Normalized Pot. Energy Def. (x-axis)
            RES_F1_normalized[class_indices, 1],  # Normalized Irrigation Def. (y-axis)
            RES_F1_normalized[class_indices, 0],  # Normalized Dev. from Natural Flow (z-axis)
            alpha=0.8,
            c=color_1_hues[i],
            marker="o",
            s=15,
            edgecolors="none",
            label=f"NSGA-II" if i == 0 else "",
        )

    for i, class_indices in enumerate(classes_2):
        ax3d.scatter(
            RES_F2_normalized[class_indices, 2],  # Normalized Pot. Energy Def. (x-axis)
            RES_F2_normalized[class_indices, 1],  # Normalized Irrigation Def. (y-axis)
            RES_F2_normalized[class_indices, 0],  # Normalized Dev. from Natural Flow (z-axis)
            alpha=0.8,
            c=color_2_hues[i],
            marker="o",
            s=15,
            edgecolors="none",
            label=f"NSGA-III" if i == 0 else "",
        )

    ax3d.set_xlabel("Enerji Üretim Açığı", labelpad=15)
    ax3d.set_ylabel("Sulama Açığı", labelpad=15)
    ax3d.set_zlabel("Doğal Akımdan Sapma", labelpad=15)

    # Set explicit limits for normalized axes
    ax3d.set_xlim(-0.05, 1.05)  # Small buffer for normalized range
    ax3d.set_ylim(-0.05, 1.05)  # Small buffer for normalized range
    ax3d.set_zlim(-0.05, 1.05)  # Small buffer for normalized range

    # ax3d.legend()
    plt.tight_layout()

    # Generate frames for the GIF
    filenames = []
    for i in range(num_frames):
        ax3d.view_init(elev=20, azim=i * rotation_speed)
        filename = f'frame_{i:03d}.png'
        plt.savefig(filename, dpi=150)
        filenames.append(filename)

    # Create the GIF
    with imageio.get_writer(output_gif_name, mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up the individual frame files
    for filename in set(filenames):
        import os
        os.remove(filename)

    plt.close(fig)
    print(f"GIF '{output_gif_name}' created successfully!")


def BCKplot_2d_pareto_compare(RES_F1, RES_F2):
    fig = plt.figure(figsize=(18, 6))  # Adjust figure size as needed
    fig.subplots_adjust(wspace=0.3)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)  # 1 row, 3 columns
        ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
        ax.scatter(
            RES_F1[:, (i + 1) % 3],
            RES_F1[:, (i + 2) % 3],
            s=15,
            alpha=0.7,
            marker="o",
            color=color_1,
            label="NSGA-II",
            edgecolors="none",
        )
        ax.scatter(
            RES_F2[:, (i + 1) % 3],
            RES_F2[:, (i + 2) % 3],
            s=15,
            alpha=0.7,
            marker="o",
            color=color_2,
            label="NSGA-III",
            edgecolors="none",
        )
        ax.set_xlabel(
            [
                "Deviation from Natural Flow",
                "Deficit from the Potential Energy",
                "Irrigation Deficits",
            ][(i + 1) % 3]

        )
        ax.set_ylabel(
            [
                "Deficit from the Potential Energy",
                "Irrigation Deficits",
                "Deviation from Natural Flow",
            ][(i + 2) % 3]

        )
        # ax.legend(fontsize=12)
    # plt.title("2D Pareto Front Comparison")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_2d_pareto_compare(RES_F1, RES_F2, color_1_hues, color_2_hues):
    """
    Plots 2D scatter plots of two Pareto fronts, with each front divided into 3 classes,
    and allows user to define the color hues. Classifies each plot based on a specific objective.
    Normalizes axis values using Min-Max scaling.
    """
    plt.rcParams["font.size"] = 11
    fig = plt.figure(figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    # --- Normalization Step ---
    combined_data = np.vstack((RES_F1, RES_F2))
    min_vals = np.min(combined_data, axis=0)
    max_vals = np.max(combined_data, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    RES_F1_normalized = (RES_F1 - min_vals) / range_vals
    RES_F2_normalized = (RES_F2 - min_vals) / range_vals
    # --- End Normalization Step ---

    def classify_data(data, axis_index):
        sorted_indices = np.argsort(data[:, axis_index])
        classes = np.array_split(sorted_indices, 3)
        return classes

    # Define objective labels for the axes using Turkish terms
    # IMPORTANT: Ensure the order here matches the actual objective order in your RES_F1/RES_F2 arrays
    # (e.g., index 0 for Dev. from Natural Flow, index 1 for Pot. Energy Def., index 2 for Irrigation Def.)
    objective_labels = [
        "Doğal Akımdan Sapma",
        "Enerji Üretim Açığı",
        "Sulama Açığı"
    ]

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)

        if i == 0:  # Plot 1: Classify based on Pot. Energy Def. (index 1)
            classes_1 = classify_data(RES_F1_normalized, 1)
            classes_2 = classify_data(RES_F2_normalized, 1)
        elif i == 1:  # Plot 2: Classify based on Irrigation Def. (index 2)
            classes_1 = classify_data(RES_F1_normalized, 2)
            classes_2 = classify_data(RES_F2_normalized, 2)
        elif i == 2:  # Plot 3: Classify based on Dev. from Natural Flow (index 0)
            classes_1 = classify_data(RES_F1_normalized, 0)
            classes_2 = classify_data(RES_F2_normalized, 0)

        x_obj_idx = (i + 1) % 3
        y_obj_idx = (i + 2) % 3

        for j, class_indices in enumerate(classes_1):
            ax.scatter(
                RES_F1_normalized[class_indices, x_obj_idx],
                RES_F1_normalized[class_indices, y_obj_idx],
                s=12, alpha=0.8, marker="o",
                color=color_1_hues[j],
                label=f"NSGA-II" if i == 0 and j == 0 else "",
                edgecolors="none",
            )

        for j, class_indices in enumerate(classes_2):
            ax.scatter(
                RES_F2_normalized[class_indices, x_obj_idx],
                RES_F2_normalized[class_indices, y_obj_idx],
                s=12, alpha=0.8, marker="o",
                color=color_2_hues[j],
                label=f"NSGA-III" if i == 0 and j == 0 else "",
                edgecolors="none",
            )

        ax.set_xlabel(objective_labels[x_obj_idx])
        ax.set_ylabel(objective_labels[y_obj_idx])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # --- MODIFIED LINES HERE ---
        ax.set_xlim(-0.05, 1.05) # Added a small buffer below 0 and above 1
        ax.set_ylim(-0.05, 1.05) # Added a small buffer below 0 and above 1
        # --- END MODIFIED LINES ---

        # --- Virgül kullanarak ondalık gösterim ---
        formatter = FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ','))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def parallel_coordinates_plot(data1, data2, color_1_hues, color_2_hues):
    """
    Generates a parallel coordinates plot with smoothed lines and legend for two datasets.

    Args:
        data1: A NumPy array containing the first dataset (e.g., NSGA-II results).
        data2: A NumPy array containing the second dataset (e.g., NSGA-III results).
        color_1_hues: A list of colors for dataset 1.
        color_2_hues: A list of colors for dataset 2.
    """
    plt.rcParams["font.size"] = 20  # Adjust this value as needed

    assert data1.shape[1] == data2.shape[1], "Datasets must have the same number of features"

    # Combine data for normalization
    data = np.vstack((data1, data2))

    # Define feature names
    ynames = ['Enerji Üretim Açığı', 'Sulama Açığı', 'Referans Akımdan Sapma']

    # Normalize the data
    ymins = data.min(axis=0)
    ymaxs = data.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # Add 5% padding
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # Scale data between 0 and 1
    zs1 = (data1 - ymins) / dys
    zs2 = (data2 - ymins) / dys

    # Create figure
    fig, host = plt.subplots(figsize=(20, 5))
    axes = [host] + [host.twinx() for _ in range(data.shape[1] - 1)]

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (data.shape[1] - 1)))

    host.set_xlim(0, data.shape[1] - 1)
    host.set_xticks(range(data.shape[1]))
    host.set_xticklabels(ynames, fontsize=10)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)

    def classify_data(data, weights=None):
        """
        Classifies data into 3 groups based on a general score.

        Args:
            data: A NumPy array where rows are solutions and columns are features.
            weights: A NumPy array of feature weights. If None, equal weighting is used.

        Returns:
            A list of 3 arrays, each containing indices of data points in a class.
        """
        if weights is None:
            weights = np.ones(data.shape[1]) / data.shape[1]  # Equal weighting

        # Compute the general score (weighted sum)
        scores = np.dot(data, weights)

        # Sort based on the computed score
        sorted_indices = np.argsort(scores)

        # Split into 3 equal groups
        return np.array_split(sorted_indices, 3)

    classes_1 = classify_data(data1)
    classes_2 = classify_data(data2)

    # Function to plot dataset lines with correct layering
    def plot_dataset(data, zs, classes, color_hues):
        # Plot the second and third classes first (to keep the first class on top)
        for j in range(1, 3):
            for k in classes[j]:
                verts = list(zip(range(data.shape[1]), zs[k, :]))
                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
                path = Path(verts, codes)
                patch = patches.PathPatch(
                    path, facecolor='none', lw=1, alpha=0.2, edgecolor=color_hues[j]
                )
                host.add_patch(patch)

        # Plot the first class (j=0) last, so it appears on top
        for k in classes[0]:
            verts = list(zip(range(data.shape[1]), zs[k, :]))
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path, facecolor='none', lw=1, alpha=0.5, edgecolor=color_hues[0]
            )
            host.add_patch(patch)

    # Plot dataset 1 (NSGA-II)
    plot_dataset(data1, zs1, classes_1, color_1_hues)

    # Plot dataset 2 (NSGA-III)
    plot_dataset(data2, zs2, classes_2, color_2_hues)

    # Create legend
    legend_patches = [
        patches.Patch(color=color_1_hues[0], label="NSGA-II Class 1"),
        patches.Patch(color=color_1_hues[1], label="NSGA-II Class 2"),
        patches.Patch(color=color_1_hues[2], label="NSGA-II Class 3"),
        patches.Patch(color=color_2_hues[0], label="NSGA-III Class 1"),
        patches.Patch(color=color_2_hues[1], label="NSGA-III Class 2"),
        patches.Patch(color=color_2_hues[2], label="NSGA-III Class 3")
    ]

    # host.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fancybox=True)

    # fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, hspace=0.9)

    # fig.patch.set_linewidth(1)
    # fig.patch.set_edgecolor('black')
    # --- Virgül kullanarak ondalık biçimlendirme ---
    formatter = FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ','))
    for ax in axes:
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()

def compromise_programming(X, F, filename="EXP_COMP_PROG_SCE.csv"):
    """
    Applies compromise programming to the Pareto front, visualizes the results, and calculates
    ENG_TOT, IRG_TOT, AVE_REG_RATIO for the selected solution.
    """

    weights = np.array([0.33, 0.33, 0.34])
    decomp = ASF()

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    # Find the best solution
    i = decomp.do(nF, 1 / weights).argmin()  # Pass ideal_point as keyword argument

    # Best overall solution
    best_solution = F[i]

    # Calculate ENG_TOT, IRG_TOT, AVE_REG_RATIO for the best solution
    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    K_1, K_2, K_3, K_SC = coef_values[0]
    DEC_VARS = np.array([float(val) for val in X[i]]).reshape(13, 12)
    RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
    (
        ENR_OBJ,
        IRG_OBJ,
        ECO_OBJ,
        ENG_TOT,
        IRG_TOT,
        TOT_ECO_DEV,
        AVE_REG_RATIO,
        SPILLWAY,
        GEN_WATER,
        IRG_WATER,
        GEN_EN_GWH,
        POWER_MW,
        POW_MW_AVE,
        STORAGES,
        RES_EVA,
        RES_ELEV,
        OUTFLOW,
        PEN_MIN_SQTOT,
        PEN_MAX_SQTOT,
        PEN_END_SQTOT,
        PEN_ALL_TOTAL,
        STO_DIF,
        IRG_DEF_SQTOT,
        PEN_SCALE,
    ) = RESULT

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], alpha=0.2, facecolor="grey", edgecolor="grey", label="All Points")
    ax.scatter(F[i, 0], F[i, 1], F[i, 2], alpha=0.6, facecolor="red", edgecolor="red", label="Compromise Programming")
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.legend()
    plt.show()

    # Print the results
    print(f"Best solution using Compromise Programming:")
    print(f"  ENG_TOT: {ENG_TOT:.4f}, IRG_TOT: {IRG_TOT:.4f}, AVE_REG_RATIO: {AVE_REG_RATIO:.4f}")

    # Export results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "ENG_TOT", "IRG_TOT", "AVE_REG_RATIO"])
        writer.writerow(["Compromise Programming", f"{ENG_TOT:.4f}", f"{IRG_TOT:.4f}", f"{AVE_REG_RATIO:.4f}"])

    # Open the exported CSV file
    os.startfile(filename)


from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder  # Import necessary classes


def corrected_do(self, F, **kwargs):
    # Multi-Criteria Decision Making - High Trade-off Points
    # Monkey patch the _do method of HighTradeoffPoints
    n, m = F.shape

    neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

    mu = np.full(n, - np.inf)

    # for each solution in the set calculate the least amount of improvement per unit deterioration
    for i in range(n):
        # for each neighbour in a specific radius of that solution
        neighbors = neighbors_finder.find(i)

        # calculate the trade-off to all neighbours
        diff = F[neighbors] - F[i]

        # calculate sacrifice and gain
        sacrifice = np.maximum(0, diff).sum(axis=1)
        gain = np.maximum(0, -diff).sum(axis=1)

        # warnings.filterwarnings("ignore")  # Remove np.
        tradeoff = sacrifice / gain

        # otherwise find the one with the smalled one
        mu[i] = np.nanmin(tradeoff)

    return find_outliers_upper_tail(mu)


def BCKline_gemplot_env_management_flow(SIM_OUTPUT_LIST, OBS_DATA, NODE_NAMES):
    # scenario_colors = ["darkgray", "#045275", "#089099", "#7ccba2", "#f0746e", "#dc3977", "#7c1d6f"]
    scenario_colors = ["#0077be", "#007A1F", "#74c476", "#FFA500", "#fdae61", "#f46d43", "#d73027"]  # Blue for observed, green-orange-red for A-F
    scenario_labels = ["Observed", "A: Natural", "B: Slightly Modified", "C: Moderately Modified", "D: Largely Modified",
                       "E: Seriously Modified", "F: Critically Modified"]

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    axs = axs.flatten()

    # Determine the overall y-axis limits with padding
    # y_min = 0 #np.min([np.min(data) for data in SIM_OUTPUT_LIST] + [np.min(OBS_DATA)]) * 0.95
    # y_max = 400 #np.max([np.max(data) for data in SIM_OUTPUT_LIST] + [np.max(OBS_DATA)]) * 1.05

    for node_index, node_name in enumerate(NODE_NAMES):
        ax = axs[node_index]

        smoothed_months = np.linspace(1, 12, 90)  # More descriptive variable name

        obs = OBS_DATA[node_index, :]
        interpolation_fn = interp1d(np.arange(1, 13), obs, kind='cubic')
        obs_smooth = interpolation_fn(smoothed_months)

        ax.plot(smoothed_months, obs_smooth, color=scenario_colors[0], linestyle="--", linewidth=1.2, alpha=0.8, zorder=5)
        # ax.plot(np.arange(1, 13), obs, color=scenario_colors[0], linestyle="--", linewidth=1, label=scenario_labels[0], zorder=5)
        ax.scatter(np.arange(1, 13), obs, color=scenario_colors[0], label=scenario_labels[0], zorder=5, s=20)
        # ax.set_ylim(y_min, y_max)
        ax.grid(True, linewidth=0.5, color='lightgrey')  # Added grid

        # Plot simulated data with smoothed lines
        for i, sim_output in enumerate(SIM_OUTPUT_LIST):
            node_data = sim_output[node_index, :]
            interpolation_fn = interp1d(np.arange(1, 13), node_data, kind='cubic')
            smoothed_outflow = interpolation_fn(smoothed_months)  # More descriptive variable name

            ax.plot(smoothed_months, smoothed_outflow, color=scenario_colors[i + 1], linewidth=1.2, alpha=0.8, zorder=3)
            # ax.plot(np.arange(1, 13), node_data, color=scenario_colors[i + 1], linewidth=1, alpha=0.8, label=scenario_labels[i + 1])
            ax.scatter(np.arange(1, 13), node_data, color=scenario_colors[i + 1], alpha=0.9, label=scenario_labels[i + 1], s=15, zorder=4)

            ax.set_title(f"{node_name}")
            ax.grid(True, linewidth=0.5, color='lightgrey')  # Added grid

        # X-axis labels and ticks
        if node_index >= 4:
            ax.set_xticks(np.arange(1, 13))
            ax.set_xticklabels(month_labels)
        else:
            ax.set_xticks([])
            ax.grid(True, linewidth=0.5, color='lightgrey')  # Added grid

        # Y-axis label
        if node_index == 0 or node_index == 4:
            ax.set_ylabel('Streamflow (hm³)')
        else:
            ax.set_ylabel("")

        # Legend in the first subplot
        if node_index == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.98))  # Adjusted position

    plt.tight_layout()
    plt.show()


def plot_reservoir_volume(STORAGES, OBS_STO, S_MAX, S_0, S_MIN, VERSION, FOOT_NOTE):
    fontP = FontProperties()
    fontP.set_size("x-small")

    ENG_RES_COUNT = 5
    ENG_NODES = [4, 5, 7, 8, 9]

    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")

    fig = plt.figure(figsize=(16, 9))
    plt.subplots_adjust(top=0.9)
    plt.annotate(FOOT_NOTE, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize=9,
                 wrap=True)
    plt.axis('off')

    fig.suptitle(VERSION, fontsize=16, fontweight="bold")

    for order, num in enumerate(ENG_NODES, start=1):
        qq = num - 1
        ax = fig.add_subplot(1, ENG_RES_COUNT, order)
        sim_s = STORAGES[qq, :]
        obs_s = OBS_STO[qq, :]
        ax.plot(months, obs_s, color="tab:blue", label="OBS REZ HACIM (HM3)")
        ax.plot(months, sim_s, color="tab:red", label="SIM REZ HACIM (HM3)")
        ax.fill_between(
            months,
            S_MIN_SERIES[qq, :],
            S_MAX_SERIES[qq, :],
            color="tab:blue",
            alpha=0.2,
        )
        ax.set_title(title[qq])
        ax.set_xlabel("AY")
        ax.grid(color="grey", linestyle=":", linewidth=0.5)
        plt.axhline(S_MAX[qq], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.axhline(S_0[qq], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.axhline(S_MIN[qq], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.text(
            2,
            S_MIN[qq],
            "S_MIN",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )
        plt.text(
            2,
            S_0[qq],
            "S_INI",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )
        plt.text(
            2,
            S_MAX[qq],
            "S_MAX",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop=fontP)

    plt.tight_layout()
    # plt.show()


def plot_reservoir_levels(OBS_ELEV, RES_ELEV, S_MAX_LEVEL, S_MIN_LEVEL):
    fontP = FontProperties()
    fontP.set_size("x-small")

    ENG_RES_COUNT = 5
    ENG_NODES = [4, 5, 7, 8, 9]
    tot_exen = ["-", "OBS", "SIM", "-"]
    tot_exen_2 = ["-", "OBS", "NAT", "SIM", "-"]

    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("REZERVUAR SEVIYELERI (M)", fontsize=16, fontweight="bold")
    for order, num in enumerate(ENG_NODES, start=1):
        qq = num - 1
        ax = fig.add_subplot(1, ENG_RES_COUNT, order)
        sim_s = RES_ELEV[qq, :]
        obs_s = OBS_ELEV[qq, :]
        ax.plot(months, obs_s, color="tab:blue", label="OBS REZ KOT (M)")
        ax.plot(months, sim_s, color="tab:red", label="SIM REZ KOT (M)")
        ax.set_title(title[qq])
        ax.set_xlabel("AY")
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
        plt.axhline(S_MAX_LEVEL[qq], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.axhline(S_MIN_LEVEL[qq], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.text(
            2,
            S_MIN_LEVEL[qq],
            "MIN",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )
        plt.text(
            2,
            S_MAX_LEVEL[qq],
            "MAX",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop=fontP)

    plt.tight_layout()
    # plt.show()


def plot_reservoir_net_evaporation(OBS_EVA, RES_EVA):
    fontP = FontProperties()
    fontP.set_size("x-small")

    ENG_RES_COUNT = 5
    ENG_NODES = [4, 5, 7, 8, 9]

    tot_exen = ["-", "OBS", "SIM", "-"]
    tot_exen_2 = ["-", "OBS", "NAT", "SIM", "-"]
    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("REZERVUAR BUHARLAŞMA MİKTARLARI (HM3)", fontsize=16, fontweight="bold")
    bar_width = 0.35  # Width of each bar
    for order, num in enumerate(ENG_NODES, start=1):
        qq = num - 1
        ax = fig.add_subplot(1, ENG_RES_COUNT, order)
        index = np.arange(len(months))  # x-axis positions for the bars
        sim_s = RES_EVA[qq, :]
        obs_s = OBS_EVA[qq, :]
        ax.bar(index - bar_width / 2, obs_s, color="tab:blue", label="OBS REZ EVA (HM3)", width=bar_width, alpha=0.9)
        ax.bar(index + bar_width / 2, sim_s, color="tab:red", label="SIM REZ EVA (HM3)", width=bar_width, alpha=0.9)
        ax.set_title(title[qq])
        ax.set_xlabel("AY")
        ax.set_xticks(index)
        ax.set_xticklabels(months)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop=fontP)

    plt.tight_layout()
    # plt.show()


def plot_spillway(SPILLWAY, SPILLAGE_CAP):
    fontP = FontProperties()
    fontP.set_size("x-small")

    ENG_RES_COUNT = 5
    ENG_NODES = [4, 5, 7, 8, 9]

    title = get_data_string("CH_NAMES")
    months = get_data_string("CH_MONTHS")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("SAVAK (HM3)", fontsize=16, fontweight="bold")
    for order, num in enumerate(ENG_NODES, start=1):
        qq = num - 1
        ax = fig.add_subplot(1, ENG_RES_COUNT, order)
        sim_splg = SPILLWAY[qq, :]
        ax.plot(months, sim_splg, color="tab:red", label="SIM SPILLAGE (HM3)")
        ax.set_title(title[qq])
        ax.set_xlabel("AY")
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
        plt.axhline(SPILLAGE_CAP[qq, 0], color="tab:orange", linestyle="--", linewidth=0.8)
        plt.text(
            2,
            SPILLAGE_CAP[qq, 0],
            "SPILLAGE CAP",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="tab:orange",
        )
        plt.grid(color="grey", linestyle=":", linewidth=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop=fontP)

    # plt.tight_layout()
    # plt.show()


def get_data(file_name):
    csv_name_path = PATHWAY / (file_name + ".csv")
    data = np.genfromtxt(csv_name_path, delimiter=",", skip_header=1)
    # Check if the array has only one dimension
    if data.ndim == 1:
        # For one-dimensional array, remove the first element
        data = data[1:].reshape(1, -1)
    else:
        # For two-dimensional array, skip the first column
        data = data[:, 1:]

    return data

def get_data_hist(file_name):
    csv_name_path = PATHWAY / "NEW_HIST_FILES" /  (file_name + ".csv")
    data = np.genfromtxt(csv_name_path, delimiter=",", skip_header=1)
    # Check if the array has only one dimension
    if data.ndim == 1:
        # For one-dimensional array, remove the first element
        data = data[1:].reshape(1, -1)
    else:
        # For two-dimensional array, skip the first column
        data = data[:, 1:]

    return data

def get_data_metrichist(file_name):
    csv_name_path = PATHWAY / "METRIC_HIST" /  (file_name + ".csv")
    data = np.genfromtxt(csv_name_path, delimiter=",", skip_header=1)
    # Check if the array has only one dimension
    if data.ndim == 1:
        # For one-dimensional array, remove the first element
        data = data[1:].reshape(1, -1)
    else:
        # For two-dimensional array, skip the first column
        data = data[:, 1:]

    return data


def get_data_string(file_name):
    csv_name_path = PATHWAY / (file_name + ".csv")

    # --- FIX: Apply robust encoding to handle Turkish characters ---
    try:
        # 1. Try the most likely non-UTF-8 encoding for Turkish files
        df = pd.read_csv(csv_name_path, encoding='windows-1254')
    except UnicodeDecodeError:
        try:
            # 2. Try a more general alternative
            df = pd.read_csv(csv_name_path, encoding='latin-1')
        except:
            # 3. Fallback to UTF-8 with BOM handling
            df = pd.read_csv(csv_name_path, encoding='utf-8-sig')
            # --- END FIX ---

    first_column_str = df.iloc[:, 1].astype(str).tolist()
    return first_column_str


def get_x_initials(dec_vars_ini):
    clipped_array = np.clip(dec_vars_ini, 0, None)  # Clip negative values to zero
    flattened_array = np.ravel(clipped_array)  # Flatten the array
    return flattened_array


def get_dec_var_best(file_name):
    RES_X = get_data(file_name)
    DEC_VARS_0 = RES_X[0]
    DEC_VARS_BEST = np.array(DEC_VARS_0).reshape(13, T)
    return DEC_VARS_BEST


def get_dec_var_cust(file_name, index_no):
    RES_X = get_data(file_name)
    DEC_VARS_0 = RES_X[index_no]
    # DEC_VARS_BEST = np.array(DEC_VARS_0).reshape(EN_COUNT + IR_COUNT, T)
    DEC_VARS_BEST = np.array(DEC_VARS_0).reshape(13, T)
    return DEC_VARS_BEST


def read_penalty_coef_path(subfolder, file_name):
    # Construct file path
    file_path = os.path.join(PATHWAY, subfolder, file_name + ".csv")
    # Initialize lists to store iteration names and coefficient values
    iter_name = []
    coef_values = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            for row in reader:
                try:
                    # Parse coefficient values and add them to coef_values list KCS REV!
                    k_values = tuple(float(val) for val in row[1:5])
                    coef_values.append(k_values)

                    # Extract iteration name and add it to iter_name list
                    iteration_name = row[0].strip()
                    iter_name.append(iteration_name)
                except ValueError:
                    pass  # print(f"Skipping row {row}, non-numeric values detected.")

        # Calculate total number of iterations
        total_iter = len(coef_values)

        # Return total number of iterations, iteration names, and coefficient values
        return total_iter, iter_name, coef_values

    except FileNotFoundError:
        print(f"Error: File {file_name}.csv not found.")
        return None, None, None


def read_penalty_coef(file_name):
    # Construct file path
    file_path = PATHWAY / (file_name + ".csv")

    # Initialize lists to store iteration names and coefficient values
    iter_name = []
    coef_values = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            for row in reader:
                try:
                    # Parse coefficient values and add them to coef_values list KCS REV!
                    k_values = tuple(float(val) for val in row[1:5])
                    coef_values.append(k_values)

                    # Extract iteration name and add it to iter_name list
                    iteration_name = row[0].strip()
                    iter_name.append(iteration_name)
                except ValueError:
                    pass  # print(f"Skipping row {row}, non-numeric values detected.")

        # Calculate total number of iterations
        total_iter = len(coef_values)

        # Return total number of iterations, iteration names, and coefficient values
        return total_iter, iter_name, coef_values

    except FileNotFoundError:
        print(f"Error: File {file_name}.csv not found.")
        return None, None, None

def export_data(file_name, csv_name):
    csv_name_path = PATHWAY / ("EXP_" + csv_name + ".csv")
    df = pd.DataFrame(file_name)
    export_to_csv = df.to_csv(csv_name_path, index=True, header=True)
    return export_to_csv

def export_data_hist(file_name, csv_name):
    csv_name_path = PATHWAY / "NEW_HIST_FILES" / ("EXP_" + csv_name + ".csv")
    df = pd.DataFrame(file_name)
    export_to_csv = df.to_csv(csv_name_path, index=True, header=True)
    return export_to_csv

def export_value(file_name, csv_name):
    # Assuming PATHWAY is defined elsewhere in your code
    csv_name_path = PATHWAY / ("EXP_" + csv_name + ".csv")

    # Create a DataFrame with a single row and single column
    df = pd.DataFrame([file_name], columns=["file_name"])

    # Export the DataFrame to a CSV file
    df.to_csv(csv_name_path, index=False)  # Set index=False to avoid writing row indices

def export_value_hist(file_name, csv_name):
    # Assuming PATHWAY is defined elsewhere in your code
    csv_name_path = PATHWAY / "NEW_HIST_FILES" / ("EXP_" + csv_name + ".csv")

    # Create a DataFrame with a single row and single column
    df = pd.DataFrame([file_name], columns=["file_name"])

    # Export the DataFrame to a CSV file
    df.to_csv(csv_name_path, index=False)  # Set index=False to avoid writing row indices


def import_value_path(subfolder, csv_name):
    # Assuming PATHWAY is defined elsewhere in your code
    csv_name_path = PATHWAY / (subfolder + "\EXP_" + csv_name + ".csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_name_path)

    # Extract the single value from the DataFrame
    value = df.iloc[0, 0]  # Assuming the value is in the first row and first column

    return value


def import_value(csv_name):
    # Assuming PATHWAY is defined elsewhere in your code
    csv_name_path = PATHWAY / ("EXP_" + csv_name + ".csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_name_path)

    # Extract the single value from the DataFrame
    value = df.iloc[0, 0]  # Assuming the value is in the first row and first column

    return value

def import_value_hist(csv_name):
    # Assuming PATHWAY is defined elsewhere in your code
    csv_name_path = PATHWAY / "NEW_HIST_FILES" / ("EXP_" + csv_name + ".csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_name_path)

    # Extract the single value from the DataFrame
    value = df.iloc[0, 0]  # Assuming the value is in the first row and first column

    return value

def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours == 0:
        elaps_time_format = f"{int(minutes):02d}:{int(seconds):02d}"
    else:
        elaps_time_format = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return elaps_time_format


def open_explorer(pathway):
    # Convert to absolute path
    abs_path = Path(pathway).resolve()

    # Open File Explorer
    os.system(f"start explorer {abs_path}")


def create_func_sym_eval(RES_COUNT, T):
    # Create DEC_VARS
    DEC_VARS = sp.symarray("X", (RES_COUNT * 2 + 1, T + 1))
    DEC_VARS = np.array(DEC_VARS[1:, 1:])  # Exclude the first column

    # Create FUNC_SYM_EVAL
    FUNC_SYM_EVAL = sp.lambdify(sp.flatten(DEC_VARS), DEC_VARS.tolist(), modules="numpy")

    return DEC_VARS, FUNC_SYM_EVAL


def show_warning():
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Hibernation Active", "The computer will hibernate after the script execution.")


def BCKP_gem_plot_scatter_collect(sim_outputs, obs_data, node_names, month_names, lower_limit=None, upper_limit=None, initial=None):
    """
    Plots scatter plots for multiple models and observed data with adjusted
    month series (October to October).

    Args:
        sim_outputs (list of numpy arrays): A list of 3D numpy arrays, each representing the
                                          simulated output of a different model.
                                          Dimensions: (num_iterations, num_nodes, 12)
        obs_data (numpy array): 2D numpy array of observed data.
                                 Dimensions: (num_nodes, 12)
        node_names (numpy array): 1D numpy array of node names.
        month_names (numpy array): 1D numpy array of month names (12 months).
        lower_limit (numpy array, optional): 1D numpy array of lower limits for each node.
        upper_limit (numpy array, optional): 1D numpy array of upper limits for each node.
        initial (numpy array, optional): 1D numpy array of initial values for each node.
    """

    num_plots = len(node_names)
    fig, axs = plt.subplots(1, num_plots, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    model_colors = ["crimson", "blue", "green"]
    model_labels = ["Model 1", "Model 2", "Model 3"]

    # Adjust the observed data series
    obs_data_adjusted = adjust_data_series(obs_data[:, np.newaxis, :])[:, 0, :]
    adjusted_month_names = np.concatenate((month_names, [month_names[0]]))  # Add October to the end

    for ax, node_name in zip(axs, node_names):
        node_idx = np.where(node_names == node_name)[0][0]
        obs = obs_data_adjusted[node_idx]

        # Plot observed data (adjusted for 13 months)
        ax.scatter(np.arange(1, 14), obs, color="c", edgecolor='teal', marker='o',
                   alpha=0.9, s=50, label="Observed Mean", zorder=3)
        smo_obs = gaussian_filter1d(obs, sigma=0.3)
        ax.plot(np.arange(1, 14), smo_obs, color="c", linestyle='-', alpha=0.3,
                linewidth=3, zorder=3)

        # Plot data for each model
        for model_idx, sim_output in enumerate(sim_outputs):
            sim_output = adjust_data_series(sim_output)  # Adjust the data series
            node_data = sim_output[:, node_idx, :]
            sim = np.mean(node_data, axis=0)

            # Plot simulated mean (adjusted for 13 months)
            ax.scatter(np.arange(1, 14), sim, color=model_colors[model_idx],
                       edgecolor='black', marker='o', alpha=0.9, s=50,
                       label=model_labels[model_idx], zorder=4)
            smo_sim = gaussian_filter1d(sim, sigma=0.3)
            ax.plot(np.arange(1, 14), smo_sim, color=model_colors[model_idx],
                    linestyle='-', alpha=0.3, linewidth=3, zorder=4)

            # Simulated Distribution (adjusted for 13 months)
            for month_idx in range(13):
                month_val = node_data[:, month_idx]
                x_val = np.repeat(month_idx + 1, len(month_val))
                ax.scatter(x_val, month_val, color=model_colors[model_idx],
                           edgecolor='black', alpha=0.15)
                # Limits
        if lower_limit is not None and upper_limit is not None and initial is not None:
            # Adjust lower_limit and upper_limit
            lower_limit = np.concatenate((lower_limit, [lower_limit[0]]))  # Add October value at the end
            upper_limit = np.concatenate((upper_limit, [upper_limit[0]]))  # Add October value at the end

            ax.axhline(upper_limit[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            ax.axhline(lower_limit[node_idx], color="cadetblue", linestyle="-", linewidth=2, alpha=1, zorder=2)
            ax.axhline(initial[node_idx], color="khaki", linestyle="-", linewidth=2, alpha=0.5, zorder=1)

            # Get the upper and lower limits of the data
            data_min = min(lower_limit[node_idx])
            data_max = max(upper_limit[node_idx])
            plot_lower_limit, plot_upper_limit = ax.get_ylim()
            lower_limit_val = min(data_min, plot_lower_limit)
            upper_limit_val = max(data_max, plot_upper_limit)
            ax.axhspan(lower_limit_val, lower_limit[node_idx][0], facecolor='lightgrey', alpha=0.3)
            ax.axhspan(upper_limit[node_idx][0], upper_limit_val, facecolor='lightgrey', alpha=0.3)
            ax.set_ylim(lower_limit_val, upper_limit_val)

        ax.set_title(f'{node_name}')
        # ax.set_xlabel('Months')
        ax.set_xticks(ticks=np.arange(1, 14))  # 13 ticks for 13 months
        ax.set_xticklabels(adjusted_month_names)
        ax.grid(True)
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def plot_pareto_mesh(RES_F):
    res_x = -RES_F[:, 0]
    res_y = RES_F[:, 1]
    res_z = RES_F[:, 2]

    res_x_norm = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    res_y_norm = (res_y - np.min(res_y)) / (np.max(res_y) - np.min(res_y))
    res_z_norm = (res_z - np.min(res_z)) / (np.max(res_z) - np.min(res_z))

    average_matrix = np.mean([res_x_norm, res_y_norm, res_z_norm], axis=0)
    c_value = average_matrix

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    my_cmap = plt.get_cmap("jet_r")

    sctt = ax.scatter3D(
        res_z, res_y, res_x, alpha=0.8, c=c_value, cmap=my_cmap, marker="o"
    )

    # Connect points with lines
    ax.plot_trisurf(res_z, res_y, res_x, alpha=0.1, color='grey')

    plt.title("NSGA-III PARETO (ENERJİ - GIDA - ÇEVRE)", fontweight="bold")
    ax.set_xlabel("Doğal Akım Regulasyonu (%)", fontweight="bold")
    ax.set_ylabel("Sulama Suyu Temini (hm³)", fontweight="bold")
    ax.set_zlabel("Hidroelektrik Enerji Üretimi (GWh)", fontweight="bold")

    # Customize tick labels after plotting
    # Set a FixedLocator for each axis
    ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    ax.zaxis.set_major_locator(FixedLocator(ax.get_zticks()))

    # Customize tick labels after setting FixedLocator
    ax.xaxis.set_ticklabels([f"{x:.2f}" for x in ax.get_xticks()])
    ax.yaxis.set_ticklabels([f"{y:.2f}" for y in ax.get_yticks()])
    ax.zaxis.set_ticklabels([f"{z:,.2f}" for z in ax.get_zticks()])

    # Set tighter layout
    # plt.tight_layout()

    # Adjust spacing between axis label and axis values
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    # Increase space between axes and axes values
    ax.tick_params(pad=10)

    cbar = fig.colorbar(sctt, ax=ax, shrink=0.5, label="Genel Değerlendirme İndeksi")
    cbar.ax.yaxis.label.set_fontweight("bold")

    # Increase space between color bar and axes
    cbar.ax.tick_params(pad=10)

    # plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

def compute_hvis(hv_1, hv_2, max_evals=1000000):
    """Computes and visualizes HVIS, normalizing each model separately."""

    hv_1 = np.array(hv_1).squeeze()
    hv_2 = np.array(hv_2).squeeze()

    num_points = min(len(hv_1), len(hv_2))
    hv_1 = hv_1[:num_points]
    hv_2 = hv_2[:num_points]
    evals = np.linspace(1, max_evals, num_points)

    # Normalize EACH algorithm's HV separately
    norm_hv_1 = hv_1 / np.max(hv_1) if np.max(hv_1) != 0 else np.zeros_like(hv_1) # Handle 0 case
    norm_hv_2 = hv_2 / np.max(hv_2) if np.max(hv_2) != 0 else np.zeros_like(hv_2) # Handle 0 case

    hvis_1 = trapezoid(norm_hv_1, evals)
    hvis_2 = trapezoid(norm_hv_2, evals)

    norm_evals = evals / max_evals  # Normalized evaluations for ideal curves

    # Corrected Logistic Function (centered at 0.5, steeper slope)
    logistic_hv = 1 / (1 + np.exp(-10 * (norm_evals - 0.5))) # steeper slope
    exp_hv = 1 - np.exp(-5 * norm_evals) # steeper slope
    linear_hv = norm_evals

    ideal_hvis_logistic = trapezoid(logistic_hv, norm_evals)
    ideal_hvis_exponential = trapezoid(exp_hv, norm_evals)
    ideal_hvis_linear = trapezoid(linear_hv, norm_evals)

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(evals, norm_hv_1, label="NSGA-II", color="teal", linewidth=2)
    plt.plot(evals, norm_hv_2, label="NSGA-III", color="red", linewidth=2)
    plt.plot(evals, logistic_hv, '--', label="Ideal (Logistic)", color="blue", linewidth=2)
    plt.plot(evals, exp_hv, '--', label="Ideal (Exponential)", color="green", linewidth=2)
    plt.plot(evals, linear_hv, '--', label="Ideal (Linear)", color="purple", linewidth=2)

    # Formatting
    plt.xlabel("Function Evaluations")
    plt.ylabel("Normalized HV")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.title("Hypervolume Indicator Score (HVIS) Comparison")
    plt.show()

    # Return Results
    return {
        "HVIS_1 (NSGA-II)": hvis_1,
        "HVIS_2 (NSGA-III)": hvis_2,
        "Ideal HVIS (Logistic)": ideal_hvis_logistic,
        "Ideal HVIS (Exponential)": ideal_hvis_exponential,
        "Ideal HVIS (Linear)": ideal_hvis_linear,
    }

