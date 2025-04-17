import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from lib import simulation_model, read_penalty_coef
from problem import main
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.util.running_metric import RunningMetricAnimation
from pymoo.visualization.scatter import Scatter
from lib import export_data

np.set_printoptions(formatter={"float": lambda x: "{: .2f}".format(x)}, linewidth=150)
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "roboto"

res = main()
hist = res.history
X, F = res.opt.get("X", "F")
A = F[::10] * 1.01

##
hist_F = []
n_evals = []

for algo in hist:
    n_evals.append(algo.evaluator.n_eval)
    opt = algo.opt
    hist_F.append(opt.get("F"))

##
all_results = []
stacked_data = np.vstack(hist_F)
# Extract x and y coordinates
xs = stacked_data[:, 0]
ys = stacked_data[:, 1]
# At the end of
ALL_SIM = stacked_data
OPT_SIM = F
export_data(ALL_SIM, "ALL_SIM")
export_data(OPT_SIM, "OPT_SIM")
## 2 objectives Pareto
fps = 20  # Frames per sec
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("NSGA PARETO FRONT", fontsize=14, fontweight="bold")

ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
sctt = ax.scatter([], [], c=[], s=10, alpha=0.8, marker="o")
ax.set_xlabel("Objective 1", fontweight="bold")
ax.set_ylabel("Objective 2", fontweight="bold")
fig.colorbar(sctt, ax=ax, label="")

nfr = len(hist_F)
xs = [frame[:, 0] for frame in hist_F]
ys = [frame[:, 1] for frame in hist_F]

# Get min and max for default axis
x_min = np.array([np.min(x) for x in xs])
x_max = np.array([np.max(x) for x in xs])
y_min = np.array([np.min(y) for y in ys])
y_max = np.array([np.max(y) for y in ys])

x_min_all = np.min(x_min)
x_max_all = np.max(x_max)
y_min_all = np.min(y_min)
y_max_all = np.max(y_max)

margin = 0.1  # Adjust the margin as needed (e.g., 0.1 for 10% margin)
x_margin = (x_max - x_min) * margin
y_margin = (y_max - y_min) * margin
x_min -= x_margin
x_max += x_margin
y_min -= y_margin
y_max += y_margin

x_margin_all = (x_max_all - x_min_all) * margin
y_margin_all = (y_max_all - y_min_all) * margin
x_min_all -= x_margin_all
x_max_all += x_margin_all
y_min_all -= y_margin_all
y_max_all += y_margin_all


def update(ifrm, xa, ya):
    ax.clear()
    ax.set_xlim(x_min[ifrm], x_max[ifrm])
    ax.set_ylim(y_min[ifrm], y_max[ifrm])
    ax.grid(color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    ax.set_xlabel("Objective 1", fontweight="bold")
    ax.set_ylabel("Objective 2", fontweight="bold")
    ax.set_title(f"Iteration {ifrm}", fontweight="bold")

    # Plot previous iterations in gray
    for i in range(ifrm):
        ax.scatter(xa[i], ya[i], c='lightgray', s=20, alpha=0.2, marker="o")

    # Calculate normalized values for each objective
    res_x = xa[ifrm]
    res_y = ya[ifrm]
    res_x_norm = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    res_y_norm = (res_y - np.min(res_y)) / (np.max(res_y) - np.min(res_y))
    average_score = (res_x_norm + res_y_norm) / 2
    my_cmap = plt.cm.viridis

    # Scatter plot
    sctt = ax.scatter(xa[ifrm], ya[ifrm], c=average_score, cmap=my_cmap, s=20, alpha=0.8)
    ax.set_title(f"Iteration {ifrm}", fontweight="bold")

    return sctt


ani = animation.FuncAnimation(fig, update, frames=nfr, fargs=(xs, ys), interval=1000 / fps)
plt.show()

## Performance Indicator - Hypvervolume (HV)
hist_F = []
n_evals = []

for algo in hist:
    n_evals.append(algo.evaluator.n_eval)
    opt = algo.opt
    hist_F.append(opt.get("F"))

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

metric = Hypervolume(ref_point=np.array([0, 2800, 1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

hv = [metric.do(_F) for _F in hist_F]

plt.figure(figsize=(7, 5))
plt.plot(n_evals, hv, color='gray', lw=0.8)
plt.scatter(n_evals, hv, facecolor="none", edgecolor='blue', marker="o", s=5, alpha=0.7)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()

## Performance Indicator - Running Metric
running = RunningMetricAnimation(delta_gen=50,
                                 n_plots=4,
                                 key_press=True,
                                 do_show=True)

for algorithm in res.history:
    running.update(algorithm)

## Inverted Generational Distance/Plus (IGD/+)
ind = IGD(F)
ind = IGDPlus(F)
print("IGD", ind(A))
print("IGD+", ind(A))

# metric = IGD(F, zero_to_one=True
metric = IGDPlus(F, zero_to_one=True)
igd = [metric.do(_F) for _F in hist_F]

plt.plot(n_evals, igd, color='gray', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, igd, facecolor="none", edgecolor='blue', marker="o", s=5, alpha=0.7)
plt.axhline(10 ** -2, color="red", label="10^-2", linestyle='dotted')
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("IGD/+")
plt.yscale("log")
plt.legend()
plt.show()

## Multi-Criteria Decision Making - Masking Penalties
total_iter, iter_name, coef_values = read_penalty_coef("PEN_COEF")
K_1, K_2, K_3, K_SC = coef_values[0]
pf_eng, pf_irg, pf_eco, pf_pen = [], [], [], []

for dec_vars in X:
    DEC_VARS = np.array([float(val) for val in dec_vars]).reshape(13, 12)
    RESULT = simulation_model(DEC_VARS, K_1, K_2, K_3, K_SC)
    ENR_OBJ, IRG_OBJ, ECO_OBJ, ENG_TOT, IRG_TOT, ECO_AVE, REG_RATIO, SPILLWAY, \
        GEN_WATER, IRG_WATER, GEN_EN_GWH, POWER_MW, POW_MW_AVE, STORAGES, \
        RES_EVA, RES_ELEV, OUTFLOW, PEN_MIN_SQTOT, PEN_MAX_SQTOT, PEN_END_SQTOT, \
        PEN_ALL_TOTAL, STO_DIF, IRG_DEF_SQTOT, PEN_SCALE = RESULT

    pf_eng.append(ENG_TOT)
    pf_irg.append(IRG_DEF_SQTOT)
    pf_eco.append(ECO_AVE)
    #
    # pf_eng.append(ENG_TOT)
    # pf_irg.append(IRG_DEF_SQTOT)
    # pf_eco.append(ECO_AVE)

    pf_pen.append(PEN_ALL_TOTAL)  # /(K_1 + K_2 + K_3)

pf_alias = np.stack((pf_eng, pf_irg, pf_eco), axis=1)
Q10 = np.percentile(pf_pen, 1)
Q90 = np.percentile(pf_pen, 99)
mask = (pf_pen >= Q10) & (pf_pen <= Q90)

FM = pf_alias[mask]  # OBJ without Penalty
FU = pf_alias[~mask]  # OBJ with Penalty
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(FU[:, 0], FU[:, 1], FU[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="Pareto Front with Penalty")
# ax.scatter(FM[:, 0], FM[:, 1], FM[:, 2], alpha=0.5,  c=FM[:, 2],  label="Pareto Front without Penalty")
# cb = plt.colorbar(ax.scatter(FM[:, 0], FM[:, 1], FM[:, 2], c=FM[:, 1]))

##
from lib import plot_pareto_3d, plot_pareto_2d

plot_pareto_3d(FM)
plot_pareto_2d(FM)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pf_alias[:, 0], pf_alias[:, 1], pf_alias[:, 2],alpha=0.5, facecolor="blue", edgecolor='blue', label="Pareto Front w/ot Penalty")
# ax.scatter(-F[:, 0], F[:, 1], F[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="Pareto Front with Penalty")


ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_zlabel("Objective 3")
ax.legend()
plt.show()

## Multi-Criteria Decision Making - Compromise Programming
weights = np.array([0.33, 0.33, 0.33])
decomp = ASF()
I = decomp(F, weights).argmin()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F[:, 0], F[:, 1], F[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="All Points")
ax.scatter(F[I, 0], F[I, 1], F[I, 2], alpha=0.6, facecolor="red", edgecolor='red', label="Highlighted Points")

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_zlabel("Objective 3")
plt.show()

## Multi-Criteria Decision Making - Pseudo-Weights
weights_sets = [
    np.array([[1 / 3, 1 / 3, 1 / 3]]),
    np.array([[0.25, 0.25, 0.50]]),
    np.array([[0.25, 0.50, 0.25]]),
    np.array([[0.05, 0.25, 0.25]])
]

pseudo_weights_list, best_solutions = [], []

for weights in weights_sets:
    I, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)
    best_solutions.append(F[I])
    pseudo_weights_list.append(pseudo_weights)
    I = np.array(best_solutions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F[:, 0], F[:, 1], F[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="All Points")
ax.scatter(I[:, 0], I[:, 1], I[:, 2], alpha=0.6, facecolor="red", edgecolor='red', label="Highlighted Points")

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.legend()
plt.show()

for i, (solution, weights, pseudo_weights) in enumerate(zip(best_solutions, weights_sets, pseudo_weights_list), 1):
    print(f"Best solution {i} regarding weights {weights}: Point {solution} - Pseudo weights {pseudo_weights}")

## Multi-Criteria Decision Making - High Trade-off Points

dm = HighTradeoffPoints()
I = np.array(dm(F))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F[:, 0], F[:, 1], F[:, 2], alpha=0.2, facecolor="grey", edgecolor='grey', label="All Points")
ax.scatter(F[I, 0], F[I, 1], F[I, 2], alpha=0.6, facecolor="red", edgecolor='red', label="High Trade-off Points")

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.legend()
plt.show()

##

