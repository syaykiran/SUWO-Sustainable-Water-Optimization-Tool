import time
import matplotlib.pyplot as plt
import numpy as np
from visualize import visualize_main
from display import MyOutput
from lib import simulation_model, read_penalty_coef, export_data, export_value, elapsed_time, get_x_initials, show_warning, get_data
from load import NVARS, LOWER_BOUND, UPPER_BOUND, RES_COUNT, OBS_DEC_VARS, T, EN_COUNT, IR_COUNT, PATHWAY
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.core.termination import TerminateIfAny
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
import os
from pymoo.core.population import Population
import pandas as pd

# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# # Monkey patch the rank_from_fronts function
# def corrected_rank_from_fronts(fronts, n):
#     # create the rank array and set values using a high integer value
#     rank = np.full(n, 10000000, dtype=int)  # Changed 1e16 to 1000000
#     for i, front in enumerate(fronts):
#         rank[front] = i
#     return rank
# NonDominatedSorting.rank_from_fronts = corrected_rank_from_fronts  # Apply the monkey patch


class SakaryaOptimization(ElementwiseProblem):
    def __init__(self, K_1, K_2, K_3, K_SC):
        super().__init__(
            n_var=EN_COUNT * T,
            n_obj=3,
            xl=LOWER_BOUND, xu=UPPER_BOUND,
            elementwise_evaluation=True)

        self.K_1 = K_1
        self.K_2 = K_2
        self.K_3 = K_3
        self.K_SC = K_SC

    def _evaluate(self, x, out, *args, **kwargs):
        DEC_VARS_0 = np.array(x).reshape(EN_COUNT, T)
        RESULT = simulation_model(DEC_VARS_0, self.K_1, self.K_2, self.K_3, self.K_SC)

        ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, TOT_ECO_DEV, AVE_REG_RATIO, SPILLWAY, GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, \
            POW_MW_AVE, STORAGES, RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, PEN_ALL_TOTAL, STO_DIF, \
            IRG_DEF_SQTOT, PEN_SCALE = RESULT

        out["F"] = [ENR_OBJ, IRG_OBJ, ECO_OBJ]
        out["ENG_TOT"] = ENG_TOT
        out["IRG_TOT"] = IRG_TOT
        out["AVE_REG_RATIO"] = AVE_REG_RATIO

        out["PEN_MIN_SQTOT"] = PEN_MIN_SQTOT
        out["PEN_MAX_SQTOT"] = PEN_MAX_SQTOT
        out["PEN_END_SQTOT"] = PEN_END_SQTOT
        out["PEN_ALL_TOTAL"] = PEN_ALL_TOTAL
        out["PEN_SCALE"] = PEN_SCALE
        out["STO_DIF"] = STO_DIF

        out["ENR_OBJ"] = ENR_OBJ
        out["IRG_OBJ"] = IRG_OBJ
        out["ECO_OBJ"] = ECO_OBJ


def main():
    pop_size = 1000
    n_gen = 1000
    scenario_code = "NS2-NATD"  # Define the scenario name

    termination = TerminateIfAny(RobustTermination(
        DesignSpaceTermination(tol=0.01, n_skip=10), period=10),
        get_termination("n_gen", n_gen)
    )

    start_time_in = time.time()
    total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
    ref_dirs = get_reference_directions("energy", 3, n_gen, seed=1)

    all_X = []
    all_F = []

    for idx, (K_1, K_2, K_3, K_SC) in enumerate(coef_values, start=1):
        start_time = time.time()
        print(f"\n{idx}/{total_iter}\n"
              f"Running model with parameters: \n"
              f"K_MIN={K_1}, K_MAX={K_2}, K_END={K_3}, K_SC={K_SC}")

        res = minimize(
            SakaryaOptimization(K_1, K_2, K_3, K_SC),
            NSGA2(pop_size=pop_size,
                  # ref_dirs=ref_dirs,
                  # sampling=LatinHypercubeSampling()
                  # crossover=SBX(eta=15, prob=0.9),
                  # mutation=PM(eta=20),
                  # sampling=get_x_initials(OBS_DEC_VARS)
                  ),
            termination,
            verbose=True,
            # save_history=True,
            output=MyOutput(), pf=True,
            seed=1,
        )

        # Collect all X and F for final export
        all_X.append(res.X)
        all_F.append(res.F)

        export_data(res.X, f"RES_X_{scenario_code}_{idx}")
        export_data(res.F, f"RES_F_{scenario_code}_{idx}")

        # Save historical data
        for gen_num, algo in enumerate(res.history):
            pd.DataFrame(algo.opt.get("X")).to_csv(
                f"hist_X_{scenario_code}_{idx}_gen_{gen_num}.csv", index=False
            )
            pd.DataFrame(algo.opt.get("F")).to_csv(
                f"hist_F_{scenario_code}_{idx}_gen_{gen_num}.csv", index=False
            )
            with open(f"n_evals_{scenario_code}_{idx}.csv", "a") as f:
                f.write(f"{algo.evaluator.n_eval}\n")

        iteration_execution_time = elapsed_time(start_time, time.time())
        print(f"Iteration execution time: {iteration_execution_time}")

    # Convert all_X and all_F into DataFrames
    df_X = pd.DataFrame(
        np.vstack(all_X), columns=[f"X{i + 1}" for i in range(EN_COUNT * T)]
    )
    df_F = pd.DataFrame(np.vstack(all_F), columns=["ENR_OBJ", "IRG_OBJ", "ECO_OBJ"])

    # Export data to CSV
    df_X.to_csv(f"ALL_RES_X_{scenario_code}.csv", index=False)
    df_F.to_csv(f"ALL_RES_F_{scenario_code}.csv", index=False)

    total_execution_time = elapsed_time(start_time_in, time.time())
    print(f">> Total execution time: {total_execution_time}")
    export_value(total_execution_time, f"total_execution_time_{scenario_code}")
    export_value(pop_size, f"pop_size_{scenario_code}")
    export_value(n_gen, f"n_gen_{scenario_code}")

    return res


if __name__ == "__main__":
    main()
    visualize_main()
    # os.system("shutdown /h")  # Hibernate the computer after execution
