import numpy as np
import pandas as pd
from pathlib2 import Path

np.set_printoptions(formatter={"float": "{: .3f}".format}, linewidth=300)
PATHWAY = Path(".")


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


def get_data_string(file_name):
    csv_name_path = PATHWAY / (file_name + ".csv")
    df = pd.read_csv(csv_name_path)
    first_column_str = df.iloc[:, 1].astype(str).tolist()
    return first_column_str


T = 12
RES_COUNT = 13
EN_COUNT = 13 #revize et
IR_COUNT = 13 #revize et
N = 13
NVARS = (EN_COUNT + IR_COUNT) * T

MX_ENR = get_data("MX_ENR")
MX_CON = get_data("MX_CON")
MX_STO = get_data("MX_STO")
MX_CON_IRG = get_data("MX_CON_IRG")
ENG_NODES = [4, 5, 7, 8, 9]
IRG_NODES = [1, 2, 3, 6, 10, 11, 12, 13]

NAT_INFLOW = get_data("NAT_INF_NOR")
# NAT_INFLOW = get_data("NAT_INF_CLM45")
# NAT_INFLOW = get_data("NAT_INF_WET")
# NAT_INFLOW = get_data("NAT_INF_DRY")
CONS_OTHER = get_data("CONS_OTHER")
CONS_HIDR = get_data("CONS_HIDR")
INFLOW = NAT_INFLOW - (CONS_OTHER + CONS_HIDR)

# NAT_EMC = get_data("NAT_EMC_A")
NAT_EMC = get_data("NAT_EMC_B")
# NAT_EMC = get_data("NAT_EMC_C")
# NAT_EMC = get_data("NAT_EMC_D")
# NAT_EMC = get_data("NAT_EMC_E")
# NAT_EMC = get_data("NAT_EMC_F")

LOWER_BOUND_CSV = get_data("LOWER_BOUND")
UPPER_BOUND_CSV = get_data("UPPER_BOUND")
# OBS_DEMAND = get_data("OBS_DEMAND") # JUST FOR REFERENCE SIMULATION!!
# OBS_DEMAND = get_data("OBS_DEMAND") # JUST FOR REFERENCE SIMULATION!!

LOWER_BOUND = np.reshape(LOWER_BOUND_CSV, (RES_COUNT * T,))  # NVARS YAZ!!!!
UPPER_BOUND = np.reshape(UPPER_BOUND_CSV, (RES_COUNT * T,))  # NVARS YAZ!!!!

EN_DEMAND = get_data("EN_DEMAND")
IR_DEMAND = get_data("IR_DEMAND")
EN_OPER_CAP = get_data("EN_OPER_CAP")
OBS_EN_FIRM = get_data("OBS_EN_FIRM")
OBS_EN_FIRM_GWH = get_data("OBS_EN_FIRM_GWH")
OBS_EN_FIRM_MW = get_data("OBS_EN_FIRM_MW")
SPILLAGE_CAP = get_data("SPILLAGE_CAP")
OBS_EN_MIN = get_data("OBS_EN_MIN")
OBS_EN_MAX = get_data("OBS_EN_MAX")
INST_CAP = get_data("INST_CAP")
GWH_CAP = get_data("GWH_CAP")

OBS_DEC_VARS = get_data("OBS_DEC_VARSX")
OBS_NAT_FLOW = get_data("OBS_NAT_FLOW")
S_0 = get_data("S_0")
OBS_STO = get_data("OBS_STO")
OBS_SDELTA = get_data("OBS_SDELTA")
OBS_SPL = get_data("OBS_SPL")
OBS_EVA = get_data("OBS_EVA")
OBS_OUTFLOW = get_data("OBS_OUTFLOW")
OBS_ELEV = get_data("OBS_ELEV")
OBS_EN_GWH = get_data("OBS_EN_GWH")
OBS_POW_MW = get_data("OBS_POW_MW")
S_MIN_SERIES = get_data("S_MIN_SERIES")
S_MAX_SERIES = get_data("S_MAX_SERIES")
S_MIN_ALL = get_data("S_MIN_ALL")
S_MAX_ALL = get_data("S_MAX_ALL")
S_MIN = get_data("S_MIN")
S_MAX = get_data("S_MAX")

S_MIN_LEVEL = get_data("S_MIN_LEVEL")
S_MAX_LEVEL = get_data("S_MAX_LEVEL")

EVA_MM = get_data("EVA_MM")
VOL_ELV_COEF = get_data("VOL_ELV_COEF")
ELV_ARE_COEF = get_data("ELV_ARE_COEF")
S_WEI = get_data("S_WEI")

OBS_EFF = get_data("OBS_EFF")
CONVERT = 1e6 / (30 * 24 * 60 * 60)  # MCM to CMS
C_GRAVITY = 9.81
TAILWAT_LEV = get_data("TAILWAT_LEV")
TUNNEL_DIA = get_data("TUNNEL_DIA")
TUNNEL_L = get_data("TUNNEL_L")
N_HOUR = get_data("N_HOUR")
ENG_RES_COUNT = 5
ENG_NODES = [4, 5, 7, 8, 9]
CH_NAMES = get_data_string("CH_NAMES")
# CH_NAMES_TR = get_data_string("CH_NAMES_TR")
CH_MONTHS_NAME = get_data_string("CH_MONTHS_NAME")
MONTHS_NO = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
LOCAL_HEADLOSS_DEF = 0  # KABUL 2 M

# TRB_EFF = 0.900  # 0.85-0.90
# JEN_EFF = 0.960  # 0.91-0.96
# TRANS_EFF = 0.985  # 0.97-0.985
# TOT_EFF = TRB_EFF * JEN_EFF * TRANS_EFF  # 0.87 DSI
