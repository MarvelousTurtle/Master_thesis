import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from cycler import cycler
import pickle
from Code.Functions_Ising_model import *

# %% Standard formatting

default_rc = plt.rcParams.copy()
# plt.style.use("seaborn")
plt.rcdefaults()
# print(plt.style.available)

# I can multiply figsize with whatever, as long as I use the same scale=xx in Overleaf, to keep the fontsize constant. 
plt.rc("figure", dpi=300, figsize=plt.rcParams["figure.figsize"]*np.array(1.1)) 
# Using standard color sequences: cycler("color", plt.colormaps["Pastel1"].colors)
plt.rc("axes", grid=False, prop_cycle=cycler("color", ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', \
                                                      'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', \
                                                      'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']))
alphabet = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

# %% Boltzmann learning versus pseudolikelihood + diagnostics of Boltzmann learning

with open("Simulations/boltz_vs_PL_100neurons_0.02binsize_1.10.npz", "rb") as file:
    params = np.load(file)
    info = params["info"]
    neuron_indices = params["neuron_indices"]
    h_boltz, J_boltz = params["h_boltz"], params["J_boltz"]
    # M_h_boltz, M_J_boltz = params["M_h_boltz"], params["M_J_boltz"]
    # SD_h_boltz, SD_J_boltz = params["SD_h_boltz"], params["SD_J_boltz"]
    hs_boltz, Js_boltz = params["hs_boltz"], params["Js_boltz"]
    means_diff, corrs_diff = params["means_diff"], params["corrs_diff"]
    h_PL, J_PL = params["h_PL"], params["J_PL"]

nr_evals, nr_iterations = 10000, 80000
iterations = np.rint(np.linspace(0, nr_iterations - 1, nr_evals)).astype("int64")

RMS_means_diff, RMS_corrs_diff = np.mean(np.sqrt(means_diff**2), axis=1), np.mean(np.sqrt(corrs_diff**2), axis=(1, 2)) 
M_h_boltz, M_J_boltz = np.mean(hs_boltz, axis=1), np.mean(Js_boltz, axis=(1, 2))
SD_h_boltz, SD_J_boltz = np.std(hs_boltz, axis=1), np.std(Js_boltz, axis=(1, 2))
print(info)

fig, axs = plt.subplots(ncols=2, nrows=5, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 3.25)))
fig.suptitle("Convergence of Boltzman learning", x=0.5, y=0.99)
axs[0,0].plot(iterations, M_h_boltz)
axs[0,0].set_title("Mean $h$")
axs[0,0].set_title("A)", loc="left", weight="bold")
# axs[0,0].set_xlabel("Interation number")
axs[0,0].set_ylabel("$M$")
axs[1,0].plot(iterations, SD_h_boltz)
axs[1,0].set_title("Standard deviation $h$")
axs[1,0].set_title("C)", loc="left", weight="bold")
# axs[1,0].set_xlabel("Interation number")
axs[1,0].set_ylabel("$SD$")
axs[0,1].plot(iterations, M_J_boltz)
axs[0,1].set_title("Mean $J$")
axs[0,1].set_title("B)", loc="left", weight="bold")
# axs[0,1].set_xlabel("Interation number")
axs[0,1].set_ylabel("$M$")
axs[1,1].plot(iterations, SD_J_boltz)
axs[1,1].set_title("Standard deviation $J$")
axs[1,1].set_title("D)", loc="left", weight="bold")
# axs[1,1].set_xlabel("Interation number")
axs[1,1].set_ylabel("$SD$")
axs[2,0].plot(iterations, RMS_means_diff)
axs[2,0].set_title("RMS error of the firing rates")
axs[2,0].set_title("E)", loc="left", weight="bold")
axs[2,0].set_yticks(np.arange(0, np.max(RMS_means_diff), 0.2))
# axs[2,0].set_xlabel("Interation number")
axs[2,0].set_ylabel(r"$\sqrt{(\langle s_i \rangle _{data} - \langle s_i \rangle _{pair})^2}$")
axs[2,1].plot(iterations, RMS_corrs_diff)
axs[2,1].set_title("RMS error of the correlations")
axs[2,1].set_title("F)", loc="left", weight="bold")
axs[2,1].set_yticks(np.arange(0, np.max(RMS_corrs_diff), 0.2))
# axs[2,1].set_xlabel("Interation number")
axs[2,1].set_ylabel(r"$\sqrt{(\langle s_i s_j \rangle _{data} - \langle s_i s_j \rangle _{pair})^2}$")
for i in range(hs_boltz.shape[1]):
    axs[3,0].plot(iterations, hs_boltz[:, i], color="C0", alpha=0.2)
axs[3,0].set_title("$h$")
axs[3,0].set_title("G)", loc="left", weight="bold")
# axs[3,0].set_xlabel("Interation number")
axs[3,0].set_xticks(np.arange(0, nr_iterations + 1, 20000))
indices_J = np.triu_indices(hs_boltz.shape[1], 1)
for i, j in zip(indices_J[0], indices_J[1]): 
    axs[3,1].plot(iterations, Js_boltz[:, i, j], color="C0", alpha=0.01)
axs[3,1].set_title("$J$") 
axs[3,1].set_title("H)", loc="left", weight="bold")
# axs[3,1].set_xlabel("Iteration number")
axs[3,1].set_xticks(np.arange(0, nr_iterations + 1, 20000))
for i in range(means_diff.shape[1]):
    axs[4,0].plot(iterations, means_diff[:, i], color="C0", alpha=0.02)
axs[4,0].set_title(r"$\langle s_i \rangle _{data} - \langle s_i \rangle _{pair}$")
axs[4,0].set_title("I)", loc="left", weight="bold")
axs[4,0].set_xlabel("Interation number")
axs[4,0].set_xticks(np.arange(0, nr_iterations + 1, 20000))
indices_corrs = np.triu_indices(means_diff.shape[1], 1)
for i, j in zip(indices_corrs[0], indices_corrs[1]): 
    axs[4,1].plot(iterations, corrs_diff[:, i, j], color="C0", alpha=0.002)
axs[4,1].set_title(r"$\langle s_i s_j \rangle _{data} - \langle s_i s_j \rangle _{pair}$") 
axs[4,1].set_title("J)", loc="left", weight="bold")
axs[4,1].set_xlabel("Iteration number")
axs[4,1].set_xticks(np.arange(0, nr_iterations + 1, 20000))
fig.tight_layout()

means_data, corrs_data = means_corrs_data(spike_trains[:, neuron_indices]) 
means_iters = means_data[None, :] - means_diff 
corrs_iters = corrs_data[None, :, :] - corrs_diff

iters = [0, 2, int(nr_evals * 0.1), nr_evals - 1]
fig, axs = plt.subplots(ncols=2, nrows=len(iters), figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73*len(iters)*0.9)))
fig.suptitle(f"Convergence of means and correlations during Boltzmann learning", x=0.5, y=0.99)
for idx, ite in enumerate(iters):
    max_means, max_corrs = np.max((means_iters[ite], means_data)), np.max((corrs_iters[ite], corrs_data))
    min_means, min_corrs = np.min((means_iters[ite], means_data)), np.min((corrs_iters[ite], corrs_data))
    axs[idx, 0].scatter(means_data, means_iters[ite], s=4, alpha=0.3)
    axs[idx, 0].plot([min_means, max_means], [min_means, max_means], linestyle="--", color="black")
    axs[idx, 0].set_title(r"$\langle s_i \rangle$" + f" at {iterations[ite]} iterations")
    axs[idx, 0].set_xlabel(r"$\langle s_i \rangle _{data}$")
    axs[idx, 0].set_ylabel(r"$\langle s_i \rangle _{pair}$")
    axs[idx, 1].scatter(corrs_data, corrs_iters[ite], s=4, alpha=0.1)
    axs[idx, 1].plot([min_corrs, max_corrs], [min_corrs, max_corrs], linestyle="--", color="black")
    axs[idx, 1].set_title(r"$\langle s_i s_j \rangle$" + f" at {iterations[ite]} iterations")
    axs[idx, 1].set_xlabel(r"$\langle s_i s_j \rangle _{data}$")
    axs[idx, 1].set_ylabel(r"$\langle s_i s_j \rangle _{pair}$")
fig.tight_layout()

# %% Boltzmann learning versus approximations (nMF, TAP, IP, SM, PL)

approx = "PL"
approx_long = "Pseudolikelihood"

with open("Simulations/boltz_vs_approx_100neurons_0.02binsize.npz", "rb") as file:
    params = np.load(file)
    info100 = params["info"]
    # neuron_indices100 = params["neuron_indices"]
    # hs_boltz100, Js_boltz100 = params["hs_boltz"], params["Js_boltz"]
    h_boltz100, J_boltz100 = params["h_boltz"], params["J_boltz"]
    h_approx100, J_approx100 = params[f"h_{approx}"], params[f"J_{approx}"]

print(info100)
    
with open("Simulations/boltz_vs_approx_20neurons_0.02binsize.npz", "rb") as file:
    params = np.load(file)
    info20 = params["info"]
    # neuron_indices20 = params["neuron_indices"]
    # hs_boltz20, Js_boltz20 = params["hs_boltz"], params["Js_boltz"]
    h_boltz20, J_boltz20 = params["h_boltz"], params["J_boltz"]
    h_approx20, J_approx20 = params[f"h_{approx}"], params[f"J_{approx}"]
    
print(info20)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 1.43)))
fig.suptitle(f"Boltzmann learning versus {approx_long}")
diagh20, diagJ20 = np.max((np.abs(h_approx20), np.abs(h_boltz20))), np.max((np.abs(J_approx20), np.abs(J_boltz20)))
axs[0,0].scatter(h_boltz20, h_approx20, s=4, alpha=0.3)
axs[0,0].plot([-diagh20, diagh20], [-diagh20, diagh20], linestyle="--", color="black")
axs[0,0].set_title("$h$ for $N=20$")
axs[0,0].set_title("A)", loc="left", weight="bold")
# axs[0,0].set_xlabel("Boltzmann $h$")
axs[0,0].set_ylabel(f"{approx} $h$")
axs[0,1].scatter(J_boltz20, J_approx20, s=4, alpha=0.3)
axs[0,1].plot([-diagJ20, diagJ20], [-diagJ20, diagJ20], linestyle="--", color="black")
axs[0,1].set_title("$J$ for $N=20$")
axs[0,1].set_title("B)", loc="left", weight="bold")
# axs[0,1].set_xlabel("Boltzmann $J$")
axs[0,1].set_ylabel(f"{approx} $J$")
diagh100, diagJ100 = np.max((np.abs(h_approx100), np.abs(h_boltz100))), np.max((np.abs(J_approx100), np.abs(J_boltz100)))
axs[1,0].scatter(h_boltz100, h_approx100, s=4, alpha=0.3)
axs[1,0].plot([-diagh100, diagh100], [-diagh100, diagh100], linestyle="--", color="black")
axs[1,0].set_title("$h$ for $N=100$")
axs[1,0].set_title("C)", loc="left", weight="bold")
axs[1,0].set_xlabel("Boltzmann $h$")
axs[1,0].set_ylabel(f"{approx} $h$")
axs[1,1].scatter(J_boltz100, J_approx100, s=4, alpha=0.3)
axs[1,1].plot([-diagJ100, diagJ100], [-diagJ100, diagJ100], linestyle="--", color="black")
axs[1,1].set_title("$J$ for $N=100$")
axs[1,1].set_title("D)", loc="left", weight="bold")
axs[1,1].set_xlabel("Boltzmann $J$")
axs[1,1].set_ylabel(f"{approx} $J$")
fig.tight_layout()

# %% G_diff vs. Boltzmann learning iteration

with open("Simulations/Gdiff_BoltzIter_neurons20_timesteps100-400000_2.npz", "rb+") as file:
    params_quality1 = np.load(file)
    info1 = params_quality1["info"]
    G_diffs1 = params_quality1["G_diffs"]
    chosen_iters1 = params_quality1["chosen_iters"]
    M_h_boltz1, M_J_boltz1 = params_quality1["M_h_boltz"], params_quality1["M_J_boltz"]
    SD_h_boltz1, SD_J_boltz1 = params_quality1["SD_h_boltz"], params_quality1["SD_J_boltz"]
with open("Simulations/Gdiff_BoltzIter_neurons20_timesteps100-400000_new.npz", "rb+") as file:
    params_quality2 = np.load(file)
    info2 = params_quality2["info"]
    G_diffs2 = params_quality2["G_diffs"]
    chosen_iters2 = params_quality2["chosen_iters"]
    M_h_boltz2, M_J_boltz2 = params_quality2["M_h_boltz"], params_quality2["M_J_boltz"]
    SD_h_boltz2, SD_J_boltz2 = params_quality2["SD_h_boltz"], params_quality2["SD_J_boltz"]
print(info1)
print(info2)

N = 20
eta = 0.01
nr_iterations = 40000
nr_samples = [100, 400000]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
for t in range(G_diffs1.shape[0]):
    for p in range(G_diffs1.shape[1]):
        ax1.plot(chosen_iters1[t, p], G_diffs1[t, p], color=f"C{t}", alpha=0.4)
        ax2.plot(chosen_iters2[t, p], G_diffs2[t, p], color=f"C{t}", alpha=0.4)
    ax2.lines[-1].set_label(f"{nr_samples[t]} samples")
# ax1.plot(chosen_iters, G_diffs)
ax1.set_xlabel("Boltzmann learning iteration")
ax1.set_ylabel(r"$\hat{G}^\mathrm{RC} - G^\mathrm{RC}$")
ax1.set_xticks(np.arange(0, 40001, 10000))
ax1.set_title(r"Approximation of $\hat{G}^\mathrm{RC}$")
ax1.set_title("A)", loc="left", weight="bold")
ax1.hlines(0, 0, nr_iterations, linestyle="--", colors="black")
# ax2.plot(chosen_iters, G_diffs)
ax2.set_xlabel("Boltzmann learning iteration")
ax2.set_ylabel(r"$\hat{G} - G$")
ax2.set_xticks(np.arange(0, 40001, 10000))
ax2.set_title(r"Approximation of $\hat{G}$")
ax2.set_title("B)", loc="left", weight="bold")
ax2.hlines(0, 0, nr_iterations, linestyle="--", colors="black")

legend = ax2.legend(markerscale=8)
for lg in legend.legendHandles:
    lg.set_alpha(1)
fig.tight_layout()

# %% Ghat vs. G_plugin

with open("Simulations/Ghat_G_20neurons_100populations_timesteps400000.npz", "rb") as file:
    quality1 = np.load(file)
    info1 = quality1["info"]
    G1 = quality1["G"]
    G1_approx = quality1["G_approx"]
print(info1)
    
with open("Simulations/Ghat_G_20neurons_100populations_timesteps400000_new.npz", "rb") as file:
    quality2 = np.load(file)
    info2 = quality2["info"]
    G2 = quality2["Gs"]
    G2_approx = quality2["Gs_approx"]
print(info2)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
diag_max = np.max((G1, G1_approx, G2, G2_approx))
diag_min = np.min((G1, G1_approx, G2, G2_approx))
ax1.scatter(G1, G1_approx, s=24, alpha=0.4)
ax1.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
ax1.set_title(r"$\hat{G}^\mathrm{RC}$ versus $G^\mathrm{RC}$")
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel(r"$G^\mathrm{RC}$")
ax1.set_ylabel(r"$\hat{G}^\mathrm{RC}$")
ax1.set_yticks(np.arange(0.85, 0.96, 0.05))
ax1.set_xticks(np.arange(0.85, 0.96, 0.05))
ax2.scatter(G2, G2_approx, s=24, alpha=0.4)
ax2.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
ax2.set_title(r"$\hat{G}$ versus $G$")
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel("$G$")
ax2.set_ylabel(r"$\hat{G}$")
ax2.set_yticks(np.arange(0.85, 0.96, 0.05))
ax2.set_xticks(np.arange(0.85, 0.96, 0.05))

fig.tight_layout()

# %% Comparison of approximations of Z

with open("Simulations/Zapprox_samples100-20000_neurons15_populatins500.npz", "rb") as file:
    Zapprox = np.load(file)
    info = Zapprox["info"]
    Z = Zapprox["Z"]
    Zs = Zapprox["Zs"]
    Zs_hat = Zapprox["Zs_hat"]
    Zs_mean = Zapprox["Zs_mean"]
    Zs_median = Zapprox["Zs_median"]
    Zs_MostSampled = Zapprox["Zs_MostSampled"]
    
print(info)

ex1, ex2 = 29, 8 # 10, 14
hist_idx = -1
samples_start, samples_end, nr_props = 100, 20000, 100 # change based on info
data_props = np.linspace(samples_start, samples_end + 1, nr_props)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=plt.rcParams["figure.figsize"]*np.array((1.30, 0.64)))
fig.suptitle("Comparison of approximations of $Z$")
ax1.axhline(1, 0, 1, color="black", linestyle="dashed")
ax1.plot(data_props, Zs_hat[ex1] / Zs[ex1], label=r"$\hat{Z}$")
ax1.plot(data_props, Zs_mean[ex1] / Zs[ex1], label=r"$\hat{Z}_{mean}$")
ax1.plot(data_props, Zs_median[ex1] / Zs[ex1], label=r"$\hat{Z}_{median}$")
ax1.plot(data_props, Zs_MostSampled[ex1] / Zs[ex1], label=r"$\hat{Z}_{MostSampled}$")
ax1.set_title("Example 1", x=0.32)
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel("Number of samples")
ax1.set_ylabel(r"$\hat{Z}_\mathrm{approx} \, / \, Z$")
ax1.set_xticks((0, 10000, 20000))
ax1.set_yticks(np.arange(0, 2.1, 0.5))
ax2.axhline(1, 0, 1, color="black", linestyle="dashed")
ax2.plot(data_props, Zs_hat[ex2] / Zs[ex2])
ax2.plot(data_props, Zs_mean[ex2] / Zs[ex2])
ax2.plot(data_props, Zs_median[ex2] / Zs[ex2])
ax2.plot(data_props, Zs_MostSampled[ex2] / Zs[ex2])
ax2.set_title("Example 2")
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel("Number of samples")
# ax2.set_ylabel(r"$\hat{Z}_\mathrm{approx} \, / \, Z$")
ax2.set_xticks((0, 10000, 20000))
ax2.set_yticks(np.arange(0, 2.1, 0.5))
ax3.axhline(1, 0, 1, color="black", linestyle="dashed")
ax3.hist(Zs_hat[:, hist_idx] / Zs[:, hist_idx], bins=20, orientation="horizontal")
ax3.set_title(r"$\hat{Z} / Z$ " + f"after {int(data_props[hist_idx] - 1)} samples", x=0.54)
ax3.set_title("C)", loc="left", weight="bold")
ax3.set_xlabel("Number of populations")
# ax3.set_ylabel(r"$\hat{Z}_\mathrm{approx} \, / \, Z$")
ax3.set_yticks(np.arange(0, 2.1, 0.5))
fig.legend(loc=(0.22, 0.53))
fig.tight_layout()

# %% Performance for small N - summing over all states 

with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_new.pkl", "rb") as file:
    params_quality = pickle.load(file)
    
print(params_quality["Info"])

neurons = np.arange(2, 21)
binsize = 0.02
populations = 100

Gs = np.full((populations, len(neurons)), np.nan)
perturb_params = np.full((populations, len(neurons)), np.nan)
firing_rates = np.full((populations, len(neurons)), np.nan)
for n_idx, n in enumerate(neurons):
    Gs[:, n_idx] = params_quality[f"N={n}"]["Gs"]
    perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
    firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
Ns = np.tile(neurons, populations).reshape(populations, len(neurons)) 
Gs10, perturb_params10 = Gs[Ns<=10], perturb_params[Ns<=10]
Gs20, perturb_params20 = Gs[Ns>10], perturb_params[Ns>10]

fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"]*np.array(0.75))
fig.suptitle("Performance of the pairwise model up to $N=20$")
# ax.scatter(perturb_params, Gs, s=1, alpha=0.3)
ax.scatter(perturb_params10, Gs10, s=1, alpha=0.3, label=r"$2 \leq N \leq 10$")
ax.scatter(perturb_params20, Gs20, s=1, alpha=0.3, label=r"$11 \leq N \leq 20$")
ax.set_xlabel(r"$N \bar v \delta t$")
ax.set_ylabel(r"$G$")
ax.set_ylim(0., 1.1) 
legend = ax.legend(markerscale=8)
for lg in legend.legendHandles:
    lg.set_alpha(1)
print(np.sum(Gs < 0), np.sum(Gs > 1.1))

# Make M and SD lines
b = 1
bins_edges = np.arange(0, 4, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
ax.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.set_xticks(np.append(bins_edges, 4))

print(np.mean(firing_rates), np.std(firing_rates))

# %% Performance for large N - approximating Z

with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality = pickle.load(file)
    
print(params_quality["Info"])

neurons = np.arange(2, 101, 1)
binsize = 0.02
populations = 100

Gs_hat = np.full((populations, len(neurons)), np.nan)
L_ratios = np.full((populations, len(neurons)), np.nan)
perturb_params = np.full((populations, len(neurons)), np.nan)
firing_rates = np.full((populations, len(neurons)), np.nan)
for n_idx, n in enumerate(neurons):
    Gs_hat[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
    L_ratios[:, n_idx] = params_quality[f"N={n}"]["error_ratios"]
    perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
    firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
Ns = np.tile(neurons, populations).reshape(populations, len(neurons))
print(np.sum(np.isnan(Gs_hat)), np.sum(np.isnan(L_ratios))) # Only really relevant for SM

# # Adding extra data to figure 
# with open("Simulations/Ghat_PL_SomCor2_binsize0.02_neurons101-200_populations50.pkl", "rb") as file:
#     params_quality2 = pickle.load(file)
# print(params_quality2["Info"])
# neurons2 = np.arange(101, 195, 1)
# binsize2 = 0.02 
# populations2 = 50
# Gs_hat2 = np.full((populations, len(neurons2)), np.nan)
# L_ratios2 = np.full((populations, len(neurons2)), np.nan)
# perturb_params2 = np.full((populations, len(neurons2)), np.nan)
# firing_rates2 = np.full((populations, len(neurons2)), np.nan)
# for n_idx, n in enumerate(neurons2):
#     Gs_hat2[0:populations2, n_idx] = params_quality2[f"N={n}"]["Gs"] 
#     L_ratios2[0:populations2, n_idx] = params_quality2[f"N={n}"]["error_ratios"]
#     perturb_params2[0:populations2, n_idx] = params_quality2[f"N={n}"]["perturb_params"]
#     firing_rates2[0:populations2, n_idx] = perturb_params2[0:populations2, n_idx] / (n * binsize2)
# Gs_hat2[Gs_hat2 == -np.inf] = np.nan # accounting for overflow errors
# L_ratios2[np.isnan(Gs_hat2)] = np.nan # accounting for overflow errors
# # print(np.where(np.isnan(Gs_hat2[:populations2, :])))
# Gs_hat, L_ratios = np.append(Gs_hat, Gs_hat2, axis=1), np.append(L_ratios, L_ratios2, axis=1)
# perturb_params, firing_rates = np.append(perturb_params, perturb_params2, axis=1), np.append(firing_rates, firing_rates2, axis=1) 
# print(np.sum(np.isnan(Gs_hat)) - (populations - populations2) * len(neurons2), \
#       np.sum(np.isnan(L_ratios)) - (populations - populations2) * len(neurons2))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
fig.suptitle("Performance of pairwise model up to $N=100$")
ax1.scatter(perturb_params, Gs_hat, s=3, alpha=0.05)
# ax1.vlines(15, 0, 1, linestyle="--", color="black", linewidth=1.5)
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel(r"$N \bar v \delta t$")
ax1.set_ylabel(r"$\hat G$")
ax1.set_ylim(-0.02, 1.1) 
ax2.scatter(perturb_params, L_ratios, s=1, alpha=0.05)
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel(r"$N \bar v \delta t$")
ax2.set_ylabel(r"$G_L$")
ax2.set_ylim(-0.02, 1.1)
fig.tight_layout()

print(np.sum(Gs_hat < 0), np.sum(Gs_hat > 1.1))

# Make M and SD lines
b = 1
bins_edges = np.arange(0, 16, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
M_L, SD_L = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    M_L[i] = np.nanmean(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    SD_L[i] = np.nanstd(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
ax1.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax1.set_xticks(bins_edges, minor=True)
ax1.set_xticks(bins_edges[::5])
ax2.errorbar(bins_edges[:-1] + b/2, M_L, yerr=SD_L, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax2.set_xticks(bins_edges, minor=True)
ax2.set_xticks(bins_edges[::5])

# Make the zoom-in plot:
axins = ax1.inset_axes((0.65, 0.65, 0.32, 0.32), transform=ax1.transAxes)
# axins = zoomed_inset_axes(ax1, 3, loc="upper right", figsize=(1, 1))
axins.scatter(perturb_params, Gs_hat, s=1, alpha=0.05)
axins.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axins.set_xlim(0., 3.)
axins.set_ylim(0.02, 1.08)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

print(np.nanmean(firing_rates), np.nanstd(firing_rates))

# %% Performance for 20-neuron populations with different mean firing rates 

with open("Simulations/G_PL_RandFiringRate_populations5000_new.npz", "rb") as file:
    params_quality20 = np.load(file, allow_pickle=True)
    h20 = params_quality20["h"]
    J20 = params_quality20["J"]
    Gs20 = params_quality20["Gs"]
    perturb_params20 = params_quality20["perturb_params"]
    neuron_indices20 = params_quality20["neuron_indices"]
with open("Simulations/G_PL_RandFiringRate_neurons_10populations5000_new.npz", "rb") as file:
    params_quality10 = np.load(file, allow_pickle=True)
    h10 = params_quality10["h"]
    J10 = params_quality10["J"]
    Gs10 = params_quality10["Gs"]
    perturb_params10 = params_quality10["perturb_params"]
    neuron_indices10 = params_quality10["neuron_indices"]
    
fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"]*np.array(0.75))
fig.suptitle(r"Performance of the pairwise model for different $\bar{v}$")
ax.scatter(perturb_params10, Gs10, s=1, alpha=0.10, label="$N=10$")
ax.scatter(perturb_params20, Gs20, s=1, alpha=0.10, label="$N=20$")
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xlabel(r"$N \bar v \delta t$")
ax.set_ylabel("$G$")
ax.set_ylim(-0.02, 1.1) 
legend = ax.legend(markerscale=8)
for lg in legend.legendHandles:
    lg.set_alpha(1)

# Make M and SD lines
b = 1
bins_edges20 = np.arange(0, 11, b)
nr_bins20 = len(bins_edges20) - 1
M_G20, SD_G20 = np.zeros(nr_bins20), np.zeros(nr_bins20)
for i in range(nr_bins20):
    M_G20[i] = np.nanmean(Gs20[np.logical_and(perturb_params20 >= bins_edges20[i], perturb_params20 <= bins_edges20[i+1])])
    SD_G20[i] = np.nanstd(Gs20[np.logical_and(perturb_params20 >= bins_edges20[i], perturb_params20 <= bins_edges20[i+1])])
bins_edges10 = np.arange(0, 7, b)
nr_bins10 = len(bins_edges10) - 1
M_G10, SD_G10 = np.zeros(nr_bins10), np.zeros(nr_bins10)
for i in range(nr_bins10):
    M_G10[i] = np.nanmean(Gs10[np.logical_and(perturb_params10 >= bins_edges10[i], perturb_params10 <= bins_edges10[i+1])])
    SD_G10[i] = np.nanstd(Gs10[np.logical_and(perturb_params10 >= bins_edges10[i], perturb_params10 <= bins_edges10[i+1])])
ax.errorbar(bins_edges20[:-1] + b/2, M_G20, yerr=SD_G20, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.errorbar(bins_edges10[:-1] + b/2, M_G10, yerr=SD_G10, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.set_xticks(bins_edges20, minor=True)
ax.set_xticks(bins_edges20[::5])

print(np.mean(perturb_params20 / (20 * 0.02)), np.std(perturb_params20 / (20 * 0.02)))
print(np.mean(perturb_params10 / (10 * 0.02)), np.std(perturb_params10 / (10 * 0.02)))

# %% Performance for 20-neuron populations with random binsize 

with open("Simulations/G_PL_RandBinsize_populations5000_new.npz", "rb") as file:
    params_quality20 = np.load(file, allow_pickle=True)
    h20 = params_quality20["h"]
    J20 = params_quality20["J"]
    Gs20 = params_quality20["Gs"]
    binsizes20 = params_quality20["binsizes"]
    perturb_params20 = params_quality20["perturb_params"]
    neuron_indices20 = params_quality20["neuron_indices"]
with open("Simulations/G_PL_RandBinsize_neurons10_populations5000_new.npz", "rb") as file:
    params_quality10 = np.load(file, allow_pickle=True)
    h10 = params_quality10["h"]
    J10 = params_quality10["J"]
    Gs10 = params_quality10["Gs"]
    binsizes10 = params_quality10["binsizes"]
    perturb_params10 = params_quality10["perturb_params"]
    neuron_indices10 = params_quality10["neuron_indices"]
    
fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"]*np.array(0.75))
fig.suptitle(r"Performance of the pairwise model for random $\delta t$")
ax.scatter(perturb_params10, Gs10, s=1, alpha=0.10, label="$N=10$")
ax.scatter(perturb_params20, Gs20, s=1, alpha=0.10, label="$N=20$")
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xlabel(r"$N \bar v \delta t$")
ax.set_ylabel("$G$")
ax.set_ylim(-0.02, 1.1) 
legend = ax.legend(markerscale=8)
for lg in legend.legendHandles:
    lg.set_alpha(1)

# Make M and SD lines
b = 1
bins_edges20 = np.arange(0, 12, b)
nr_bins20 = len(bins_edges20) - 1
M_G20, SD_G20 = np.zeros(nr_bins20), np.zeros(nr_bins20)
for i in range(nr_bins20):
    M_G20[i] = np.nanmean(Gs20[np.logical_and(perturb_params20 >= bins_edges20[i], perturb_params20 <= bins_edges20[i+1])])
    SD_G20[i] = np.nanstd(Gs20[np.logical_and(perturb_params20 >= bins_edges20[i], perturb_params20 <= bins_edges20[i+1])])
bins_edges10 = np.arange(0, 8, b)
nr_bins10 = len(bins_edges10) - 1
M_G10, SD_G10 = np.zeros(nr_bins10), np.zeros(nr_bins10)
for i in range(nr_bins10):
    M_G10[i] = np.nanmean(Gs10[np.logical_and(perturb_params10 >= bins_edges10[i], perturb_params10 <= bins_edges10[i+1])])
    SD_G10[i] = np.nanstd(Gs10[np.logical_and(perturb_params10 >= bins_edges10[i], perturb_params10 <= bins_edges10[i+1])])
ax.errorbar(bins_edges20[:-1] + b/2, M_G20, yerr=SD_G20, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.errorbar(bins_edges10[:-1] + b/2, M_G10, yerr=SD_G10, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.set_xticks(bins_edges20, minor=True)
ax.set_xticks(bins_edges20[::5])

firing_rates10 = perturb_params10 / (10 * binsizes10)
print(np.mean(firing_rates10), np.std(firing_rates10))
firing_rates20 = perturb_params20 / (20 * binsizes20)
print(np.mean(firing_rates20), np.std(firing_rates20))

# %% Performance for random number of neurons, firing rate, and binsize

with open("Simulations/Ghat_PL_RandPopulations5000.npz", "rb") as file:
    quality = np.load(file, allow_pickle=True)
    Gs = quality["Gs"]
    perturb_params = quality["perturb_params"]
    neuron_indices = quality["neuron_indices"]
    N_vbar_deltat = quality["N_vbar_deltat"]
    
fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"]*np.array(0.85))
fig.suptitle(r"Performance of the pairwise model for random values of $N$, $\bar{v}$, and $\delta t$")
ax.scatter(perturb_params, Gs, s=2, alpha=0.10)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xlabel(r"$N \bar v \delta t$")
ax.set_ylabel(r"$\hat{G}^\mathrm{RC}$")
ax.set_ylim(0., 1.1) 

# Make M and SD lines
b = 1
bins_edges = np.arange(0, 41, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
ax.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax.set_xticks(bins_edges, minor=True)
ax.set_xticks(bins_edges[::10])

firing_rates = np.array([pp[1] for pp in N_vbar_deltat])
print(np.mean(firing_rates), np.std(firing_rates))

# %% Performance for different number of neurons, firing rate, and binsize

with open("Simulations/G_PL_binsizes3_subpopulations1500.npz", "rb") as file:
    params_qualityA = np.load(file, allow_pickle=True)
    hA = params_qualityA["h"]
    JA = params_qualityA["J"]
    GsA = params_qualityA["Gs"]
    perturb_paramsA = params_qualityA["perturb_params"]
    neuron_indicesA = params_qualityA["neuron_indices"]
    
with open("Simulations/G_PL_binsizes3_subpopulations1500_B.npz", "rb") as file:
    params_qualityB = np.load(file, allow_pickle=True)
    hB = params_qualityB["h"]
    JB = params_qualityB["J"]
    GsB = params_qualityB["Gs"]
    perturb_paramsB = params_qualityB["perturb_params"]
    neuron_indicesB = params_qualityB["neuron_indices"]
    
binsizes = [0.005, 0.01, 0.02]
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
fig.suptitle(r"Performance of the pairwise model for different $N$, $\bar{v}$, and $\delta t$")
for idx, b in enumerate(binsizes):
    ax1.scatter(perturb_paramsA[:, idx], GsA[:, idx], s=2, alpha=0.15, label=fr"$\delta t = {b}$")
    ax2.scatter(perturb_paramsB[:, idx], GsB[:, idx], s=2, alpha=0.15)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.set_xlabel(r"$N \bar v \delta t$")
ax1.set_ylabel("$G^\mathrm{RC}$")
ax1.set_title("$N=20$")
ax1.set_title("A)", loc="left", weight="bold")
ax2.set_yticks(np.arange(0.6, 1.1, 0.2))
ax2.set_xlabel(r"$N \bar v \delta t$")
ax2.set_ylabel("$G^\mathrm{RC}$")
ax2.set_title("$N=10$")
ax2.set_title("B)", loc="left", weight="bold")
fig.legend(loc=[0.09, 0.18], markerscale=7)
fig.tight_layout()

# %% Performance for large N with nMF, TAP, IP, and SM

# with open("Simulations/Ghat_nMF_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
#     params_quality_nMF = pickle.load(file)
# with open("Simulations/Ghat_TAP_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
#     params_quality_TAP = pickle.load(file)
# with open("Simulations/Ghat_IP_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
#     params_quality_IP = pickle.load(file)
# with open("Simulations/Ghat_SM_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
#     params_quality_SM = pickle.load(file)

with open("Simulations/Ghat_nMF_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality_nMF = pickle.load(file)
with open("Simulations/Ghat_TAP_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality_TAP = pickle.load(file)
with open("Simulations/Ghat_IP_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality_IP = pickle.load(file)
with open("Simulations/Ghat_SM_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality_SM = pickle.load(file)
    
print(params_quality_nMF["Info"])
print(params_quality_TAP["Info"])
print(params_quality_IP["Info"])
print(params_quality_SM["Info"])

approx_long = ("Naive mean-field", "Thouless-Anderson-Palmer", "Independent pairs", "Sessak-Monasson")

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73*3.4)))
fig.suptitle("Performance of pairwise model up to $N=100$ using nMF, TAP, IP, and SM", y=0.99)
for idx, params_quality in enumerate((params_quality_nMF, params_quality_TAP, params_quality_IP, params_quality_SM)):
    
    neurons = np.arange(2, 101, 1)
    binsize = 0.02
    populations = 100
    
    Gs_hat = np.full((populations, len(neurons)), np.nan)
    L_ratios = np.full((populations, len(neurons)), np.nan)
    perturb_params = np.full((populations, len(neurons)), np.nan)
    firing_rates = np.full((populations, len(neurons)), np.nan)
    for n_idx, n in enumerate(neurons):
        Gs_hat[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
        L_ratios[:, n_idx] = params_quality[f"N={n}"]["error_ratios"]
        perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
        firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
    print(np.sum(np.isnan(Gs_hat)), np.sum(np.isnan(L_ratios))) # Only really relevant for SM
    
    axs[idx,0].scatter(perturb_params, Gs_hat, s=1, alpha=0.05)
    axs[idx,0].annotate(f"{approx_long[idx]}", xy=(-0.23, 0.5), xycoords=axs[idx,0].transAxes, rotation="vertical", va="center", fontsize=12) 
    axs[idx,0].set_title(f"{alphabet[2*idx]})", loc="left", weight="bold")
    # axs[idx,0].set_title(f"{approx_long[idx]}")
    # axs[idx,0].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,0].set_ylabel(r"$\hat{G}$")
    axs[idx,0].set_ylim(-0.02, 1.1) 
    axs[idx,1].scatter(perturb_params, L_ratios, s=1, alpha=0.05)
    axs[idx,1].set_title(f"{alphabet[2*idx+1]})", loc="left", weight="bold")
    # axs[idx,1].set_title(f"{approx_long[idx]}")
    # axs[idx,1].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,1].set_ylabel(r"$G_L$")
    axs[idx,1].set_ylim(-0.02, 1.1)
    
    print(np.sum(Gs_hat < 0), np.sum(Gs_hat > 1.1))
    
    # Make M and SD lines
    Gs_hat[Gs_hat < 0] = np.nan
    L_ratios[L_ratios < 0] = np.nan
    b = 1
    bins_edges = np.arange(0, np.floor(np.max(perturb_params)) - 1, b)
    nr_bins = len(bins_edges) - 1
    M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
    M_L, SD_L = np.zeros(nr_bins), np.zeros(nr_bins)
    for i in range(nr_bins):
        M_G[i] = np.nanmean(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        M_L[i] = np.nanmean(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_G[i] = np.nanstd(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_L[i] = np.nanstd(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    axs[idx,0].errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,0].set_xticks(bins_edges, minor=True)
    axs[idx,0].set_xticks(bins_edges[::5])
    axs[idx,0].tick_params(labelbottom=False) 
    axs[idx,1].errorbar(bins_edges[:-1] + b/2, M_L, yerr=SD_L, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,1].set_xticks(bins_edges, minor=True)
    axs[idx,1].set_xticks(bins_edges[::5])
    axs[idx,1].tick_params(labelbottom=False) 
    
    print(np.nanmean(firing_rates), np.nanstd(firing_rates))

axs[3,0].tick_params(labelbottom=True) 
axs[3,0].set_xlabel(r"$N \bar v \delta t$")
axs[3,1].tick_params(labelbottom=True) 
axs[3,1].set_xlabel(r"$N \bar v \delta t$")
fig.tight_layout()

# %% Performance in visual and auditory cortex (same binsize)

with open("Simulations/Ghat_PL_VisCor_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_VisCor = pickle.load(file)
with open("Simulations/Ghat_PL_AudCor_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_AudCor = pickle.load(file)
with open("Simulations/Ghat_PL_AudCor_binsize0.02_neurons101-200_populations50.pkl", "rb") as file:
    params_quality_AudCor_extra = pickle.load(file)
    
print(params_quality_VisCor["Info"])
print(params_quality_AudCor["Info"])
print(params_quality_AudCor_extra["Info"])

binsize = 0.02
area = ("Visual cortex", "Auditory cortex")
extra_data = (False, True)
params_quality_full = (params_quality_VisCor, params_quality_AudCor)
params_quality_extra = (None, params_quality_AudCor_extra)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73*1.9)))
fig.suptitle("Performance of pairwise model in visual and auditory cortex", y=0.97)
for idx, params_quality in enumerate(params_quality_full):
    
    neurons = np.arange(2, 101, 1)
    populations = 100
    
    Gs_hat = np.full((populations, len(neurons)), np.nan)
    L_ratios = np.full((populations, len(neurons)), np.nan)
    perturb_params = np.full((populations, len(neurons)), np.nan)
    firing_rates = np.full((populations, len(neurons)), np.nan)
    for n_idx, n in enumerate(neurons):
        Gs_hat[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
        L_ratios[:, n_idx] = params_quality[f"N={n}"]["error_ratios"]
        perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
        firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
    print(np.sum(np.isnan(Gs_hat)), np.sum(np.isnan(L_ratios))) # Only really relevant for SM
    
    if extra_data[idx]:
        neurons2 = np.arange(101, 200, 1)
        binsize2 = 0.02 
        populations2 = 50
        Gs_hat2 = np.full((populations, len(neurons2)), np.nan)
        L_ratios2 = np.full((populations, len(neurons2)), np.nan)
        perturb_params2 = np.full((populations, len(neurons2)), np.nan)
        firing_rates2 = np.full((populations, len(neurons2)), np.nan)
        for n_idx, n in enumerate(neurons2):
            Gs_hat2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["Gs"] 
            L_ratios2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["error_ratios"]
            perturb_params2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["perturb_params"]
            firing_rates2[0:populations2, n_idx] = perturb_params2[0:populations2, n_idx] / (n * binsize2)
        Gs_hat2[Gs_hat2 == -np.inf] = np.nan # accounting for overflow errors
        L_ratios2[np.isnan(Gs_hat2)] = np.nan # accounting for overflow errors
        Gs_hat, L_ratios = np.append(Gs_hat, Gs_hat2, axis=1), np.append(L_ratios, L_ratios2, axis=1)
        perturb_params, firing_rates = np.append(perturb_params, perturb_params2, axis=1), np.append(firing_rates, firing_rates2, axis=1) 
        print(np.sum(np.isnan(Gs_hat)) - (populations - populations2) * len(neurons2), \
              np.sum(np.isnan(L_ratios)) - (populations - populations2) * len(neurons2))
    
    axs[idx,0].scatter(perturb_params, Gs_hat, s=1, alpha=0.05)
    axs[idx,0].annotate(f"{area[idx]}", xy=(-0.23, 0.5), xycoords=axs[idx,0].transAxes, rotation="vertical", va="center", fontsize=12) 
    axs[idx,0].set_title(f"{alphabet[2*idx]})", loc="left", weight="bold")
    # axs[idx,0].set_title(f"{area[idx]}")
    # axs[idx,0].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,0].set_ylabel(r"$\hat{G}^\mathrm{RC}$")
    axs[idx,0].set_ylim(-0.02, 1.1) 
    axs[idx,1].scatter(perturb_params, L_ratios, s=1, alpha=0.05)
    axs[idx,1].set_title(f"{alphabet[2*idx+1]})", loc="left", weight="bold")
    # axs[idx,1].set_title(f"{area[idx]}")
    # axs[idx,1].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,1].set_ylabel(r"$G_L^\mathrm{RC}$")
    axs[idx,1].set_ylim(-0.02, 1.1)
    
    print(np.sum(Gs_hat < 0), np.sum(Gs_hat > 1.1))
    
    # Make M and SD lines
    b = 1
    bins_edges = np.arange(0, np.floor(np.nanmax(perturb_params)), b)
    nr_bins = len(bins_edges) - 1
    M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
    M_L, SD_L = np.zeros(nr_bins), np.zeros(nr_bins)
    for i in range(nr_bins):
        M_G[i] = np.nanmean(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        M_L[i] = np.nanmean(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_G[i] = np.nanstd(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_L[i] = np.nanstd(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    axs[idx,0].errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,0].set_xticks(bins_edges, minor=True)
    axs[idx,0].set_xticks(bins_edges[::5])
    axs[idx,1].errorbar(bins_edges[:-1] + b/2, M_L, yerr=SD_L, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,1].set_xticks(bins_edges, minor=True)
    axs[idx,1].set_xticks(bins_edges[::5])
    
    print(np.nanmean(firing_rates), np.nanstd(firing_rates))

axs[1,0].set_xlabel(r"$N \bar v \delta t$")
axs[1,1].set_xlabel(r"$N \bar v \delta t$")
fig.tight_layout()

# %% Performance in motor cortex (different binsize)

with open("Simulations/Ghat_PL_MotCor_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_MotCor = pickle.load(file)
with open("Simulations/Ghat_PL_MotCor_binsize0.02_neurons101-200_populations50.pkl", "rb") as file:
    params_quality_MotCor_extra = pickle.load(file)
with open("Simulations/Ghat_PL_MotCor_binsize0.06_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_MotCor_b2 = pickle.load(file)
    
print(params_quality_MotCor["Info"])
print(params_quality_MotCor_extra["Info"])
print(params_quality_MotCor_b2["Info"])

binsizes = (0.02, 0.06)
area = (r"Motor cortex with $\delta t =$" + f" ${binsizes[0]}$", r"Motor cortex with $\delta t =$" + f" ${binsizes[1]}$")
extra_data = (True, False)
params_quality_full = (params_quality_MotCor, params_quality_MotCor_b2)
params_quality_extra = (params_quality_MotCor_extra, None)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73*1.9)))
fig.suptitle("Performance of pairwise model in motor cortex", y=0.97)
for idx, params_quality in enumerate(params_quality_full):
    
    neurons = np.arange(2, 101, 1)
    populations = 100
    
    Gs_hat = np.full((populations, len(neurons)), np.nan)
    L_ratios = np.full((populations, len(neurons)), np.nan)
    perturb_params = np.full((populations, len(neurons)), np.nan)
    firing_rates = np.full((populations, len(neurons)), np.nan)
    for n_idx, n in enumerate(neurons):
        Gs_hat[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
        L_ratios[:, n_idx] = params_quality[f"N={n}"]["error_ratios"]
        perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
        firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
    print(np.sum(np.isnan(Gs_hat)), np.sum(np.isnan(L_ratios))) # Only really relevant for SM
    
    if extra_data[idx]:
        neurons2 = np.arange(101, 200, 1)
        binsize2 = 0.02 
        populations2 = 50
        Gs_hat2 = np.full((populations, len(neurons2)), np.nan)
        L_ratios2 = np.full((populations, len(neurons2)), np.nan)
        perturb_params2 = np.full((populations, len(neurons2)), np.nan)
        firing_rates2 = np.full((populations, len(neurons2)), np.nan)
        for n_idx, n in enumerate(neurons2):
            Gs_hat2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["Gs"] 
            L_ratios2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["error_ratios"]
            perturb_params2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["perturb_params"]
            firing_rates2[0:populations2, n_idx] = perturb_params2[0:populations2, n_idx] / (n * binsize2)
        Gs_hat2[Gs_hat2 == -np.inf] = np.nan # accounting for overflow errors
        L_ratios2[np.isnan(Gs_hat2)] = np.nan # accounting for overflow errors
        Gs_hat, L_ratios = np.append(Gs_hat, Gs_hat2, axis=1), np.append(L_ratios, L_ratios2, axis=1)
        perturb_params, firing_rates = np.append(perturb_params, perturb_params2, axis=1), np.append(firing_rates, firing_rates2, axis=1) 
        print(np.sum(np.isnan(Gs_hat)) - (populations - populations2) * len(neurons2), \
              np.sum(np.isnan(L_ratios)) - (populations - populations2) * len(neurons2))
    
    axs[idx,0].scatter(perturb_params, Gs_hat, s=1, alpha=0.05)
    axs[idx,0].annotate(f"{area[idx]}", xy=(-0.23, 0.5), xycoords=axs[idx,0].transAxes, rotation="vertical", va="center", fontsize=12) 
    axs[idx,0].set_title(f"{alphabet[2*idx]})", loc="left", weight="bold")
    # axs[idx,0].set_title(f"{area[idx]}")
    # axs[idx,0].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,0].set_ylabel(r"$\hat{G}^\mathrm{RC}$")
    axs[idx,0].set_ylim(-0.02, 1.1) 
    axs[idx,1].scatter(perturb_params, L_ratios, s=1, alpha=0.05)
    axs[idx,1].set_title(f"{alphabet[2*idx+1]})", loc="left", weight="bold")
    # axs[idx,1].set_title(f"{area[idx]}")
    # axs[idx,1].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,1].set_ylabel(r"$G_L^\mathrm{RC}$")
    axs[idx,1].set_ylim(-0.02, 1.1)
    
    print(np.sum(Gs_hat < 0), np.sum(Gs_hat > 1.1))
    
    # Make M and SD lines
    b = 1
    bins_edges = np.arange(0, np.floor(np.nanmax(perturb_params)), b)
    nr_bins = len(bins_edges) - 1
    M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
    M_L, SD_L = np.zeros(nr_bins), np.zeros(nr_bins)
    for i in range(nr_bins):
        M_G[i] = np.nanmean(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        M_L[i] = np.nanmean(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_G[i] = np.nanstd(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_L[i] = np.nanstd(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    axs[idx,0].errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,0].set_xticks(bins_edges, minor=True)
    axs[idx,0].set_xticks(bins_edges[::5])
    axs[idx,1].errorbar(bins_edges[:-1] + b/2, M_L, yerr=SD_L, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,1].set_xticks(bins_edges, minor=True)
    axs[idx,1].set_xticks(bins_edges[::5])
    
    print(np.nanmean(firing_rates), np.nanstd(firing_rates))

axs[1,0].set_xlabel(r"$N \bar v \delta t$")
axs[1,1].set_xlabel(r"$N \bar v \delta t$")
fig.tight_layout()

# %% Performance in somatosensory cortex (different binsize)

with open("Simulations/Ghat_PL_SomCor2_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_SomCor = pickle.load(file)
with open("Simulations/Ghat_PL_SomCor2_binsize0.02_neurons101-200_populations50.pkl", "rb") as file:
    params_quality_SomCor_extra = pickle.load(file)
with open("Simulations/Ghat_PL_SomCor2_binsize0.14_neurons2-100_populations100.pkl", "rb") as file:
    params_quality_SomCor_b2 = pickle.load(file)
    
print(params_quality_SomCor["Info"])
print(params_quality_SomCor_extra["Info"])
print(params_quality_SomCor_b2["Info"])

binsizes = (0.02, 0.14)
area = (r"Somatosensory cortex with $\delta t =$" + f" ${binsizes[0]}$", r"Somatosensory cortex with $\delta t =$" + f" ${binsizes[1]}$")
extra_data = (True, False)
params_quality_full = (params_quality_SomCor, params_quality_SomCor_b2)
params_quality_extra = (params_quality_SomCor_extra, None)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73*1.9)))
fig.suptitle("Performance of pairwise model in somatosensory cortex", y=0.97)
for idx, params_quality in enumerate(params_quality_full):
    
    neurons = np.arange(2, 101, 1)
    populations = 100
    
    Gs_hat = np.full((populations, len(neurons)), np.nan)
    L_ratios = np.full((populations, len(neurons)), np.nan)
    perturb_params = np.full((populations, len(neurons)), np.nan)
    firing_rates = np.full((populations, len(neurons)), np.nan)
    for n_idx, n in enumerate(neurons):
        Gs_hat[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
        L_ratios[:, n_idx] = params_quality[f"N={n}"]["error_ratios"]
        perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
        firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)
    print(np.sum(np.isnan(Gs_hat)), np.sum(np.isnan(L_ratios))) # Only really relevant for SM
    
    if extra_data[idx]:
        neurons2 = np.arange(101, 195, 1)
        binsize2 = 0.02 
        populations2 = 50
        Gs_hat2 = np.full((populations, len(neurons2)), np.nan)
        L_ratios2 = np.full((populations, len(neurons2)), np.nan)
        perturb_params2 = np.full((populations, len(neurons2)), np.nan)
        firing_rates2 = np.full((populations, len(neurons2)), np.nan)
        for n_idx, n in enumerate(neurons2):
            Gs_hat2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["Gs"] 
            L_ratios2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["error_ratios"]
            perturb_params2[0:populations2, n_idx] = params_quality_extra[idx][f"N={n}"]["perturb_params"]
            firing_rates2[0:populations2, n_idx] = perturb_params2[0:populations2, n_idx] / (n * binsize2)
        Gs_hat2[Gs_hat2 == -np.inf] = np.nan # accounting for overflow errors
        L_ratios2[np.isnan(Gs_hat2)] = np.nan # accounting for overflow errors
        Gs_hat, L_ratios = np.append(Gs_hat, Gs_hat2, axis=1), np.append(L_ratios, L_ratios2, axis=1)
        perturb_params, firing_rates = np.append(perturb_params, perturb_params2, axis=1), np.append(firing_rates, firing_rates2, axis=1) 
        print(np.sum(np.isnan(Gs_hat)) - (populations - populations2) * len(neurons2), \
              np.sum(np.isnan(L_ratios)) - (populations - populations2) * len(neurons2))
    
    axs[idx,0].scatter(perturb_params, Gs_hat, s=1, alpha=0.05)
    axs[idx,0].annotate(f"{area[idx]}", xy=(-0.23, 0.5), xycoords=axs[idx,0].transAxes, rotation="vertical", va="center", fontsize=12) 
    axs[idx,0].set_title(f"{alphabet[2*idx]})", loc="left", weight="bold")
    # axs[idx,0].set_title(f"{area[idx]}")
    # axs[idx,0].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,0].set_ylabel(r"$\hat{G}^\mathrm{RC}$")
    axs[idx,0].set_ylim(-0.02, 1.1) 
    axs[idx,1].scatter(perturb_params, L_ratios, s=1, alpha=0.05)
    axs[idx,1].set_title(f"{alphabet[2*idx+1]})", loc="left", weight="bold")
    # axs[idx,1].set_title(f"{area[idx]}")
    # axs[idx,1].set_xlabel(r"$N \bar v \delta t$")
    axs[idx,1].set_ylabel(r"$G_L^\mathrm{RC}$")
    axs[idx,1].set_ylim(-0.02, 1.1)
    
    print(np.sum(Gs_hat < 0), np.sum(Gs_hat > 1.1))
    
    # Make M and SD lines
    b = 1
    bins_edges = np.arange(0, np.floor(np.nanmax(perturb_params)), b)
    nr_bins = len(bins_edges) - 1
    M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
    M_L, SD_L = np.zeros(nr_bins), np.zeros(nr_bins)
    for i in range(nr_bins):
        M_G[i] = np.nanmean(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        M_L[i] = np.nanmean(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_G[i] = np.nanstd(Gs_hat[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
        SD_L[i] = np.nanstd(L_ratios[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    axs[idx,0].errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,0].set_xticks(bins_edges, minor=True)
    axs[idx,0].set_xticks(bins_edges[::5])
    axs[idx,1].errorbar(bins_edges[:-1] + b/2, M_L, yerr=SD_L, linestyle="-", elinewidth=1.0, \
                marker="o", markersize=2.5, color="black", ecolor="black")
    axs[idx,1].set_xticks(bins_edges, minor=True)
    axs[idx,1].set_xticks(bins_edges[::5])
    
    print(np.nanmean(firing_rates), np.nanstd(firing_rates))

axs[1,0].set_xlabel(r"$N \bar v \delta t$")
axs[1,1].set_xlabel(r"$N \bar v \delta t$")
fig.tight_layout()

# %% Example of third-order correlations in pairwise model and data

with open("Simulations/corrs3_3-100neurons.pkl", "rb") as file:
    corrs3 = np.load(file, allow_pickle=True)
    
print(corrs3["Info"])

with open("Simulations/concorrs3_3-100neurons.pkl", "rb") as file:
    concorrs3 = np.load(file, allow_pickle=True)
    
print(concorrs3["Info"])

n, pop1, pop2 = 100, 2, 15

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73 * 1.9)))
fig.suptitle("Third-order correlations in data and pairwise model")

corrs3_data = corrs3[f"N={n}"]["corrs3_data"][pop1]
corrs3_pair = corrs3[f"N={n}"]["corrs3_pair"][pop1]
concorrs3_data = concorrs3[f"N={n}"]["corrs3_data"][pop1]
concorrs3_pair = concorrs3[f"N={n}"]["corrs3_pair"][pop1]

# nr_bins = 1000
# corrs3_bins_edges = np.quantile(corrs3_data, np.linspace(0, 1, nr_bins + 1, endpoint=True))
# concorrs3_bins_edges = np.quantile(concorrs3_data, np.linspace(0, 1, nr_bins + 1, endpoint=True))
# corrs3_M_data, corrs3_SD_data = np.zeros(nr_bins), np.zeros(nr_bins)
# corrs3_M_pair, corrs3_SD_pair = np.zeros(nr_bins), np.zeros(nr_bins)
# concorrs3_M_data, concorrs3_SD_data = np.zeros(nr_bins), np.zeros(nr_bins)
# concorrs3_M_pair, concorrs3_SD_pair = np.zeros(nr_bins), np.zeros(nr_bins)
# for b in range(nr_bins):
#     corrs3_mask = np.logical_and(corrs3_data > corrs3_bins_edges[b], corrs3_data < corrs3_bins_edges[b+1])
#     concorrs3_mask = np.logical_and(concorrs3_data > concorrs3_bins_edges[b], concorrs3_data < concorrs3_bins_edges[b+1])
#     corrs3_M_data[b], corrs3_SD_data[b] = np.mean(corrs3_data[corrs3_mask]), np.std(corrs3_data[corrs3_mask])
#     corrs3_M_pair[b], corrs3_SD_pair[b] = np.mean(corrs3_pair[corrs3_mask]), np.std(corrs3_pair[corrs3_mask])
#     concorrs3_M_data[b], concorrs3_SD_data[b] = np.mean(concorrs3_data[concorrs3_mask]), np.std(concorrs3_data[concorrs3_mask])
#     concorrs3_M_pair[b], concorrs3_SD_pair[b] = np.mean(concorrs3_pair[concorrs3_mask]), np.std(concorrs3_pair[concorrs3_mask])

# axs[0,0].errorbar(corrs3_M_data, corrs3_M_pair, xerr=corrs3_SD_data, yerr=corrs3_SD_pair, linestyle="None", \
#               elinewidth=0.5, marker="o", markersize=2.5, color="black", ecolor="red")
# corrs3_diag_max = np.max((corrs3_M_data, corrs3_M_pair)) + np.max((corrs3_SD_pair, corrs3_SD_data))
# corrs3_diag_min = np.min((corrs3_M_data, corrs3_M_pair)) - np.max((corrs3_SD_pair, corrs3_SD_data))
axs[0,0].scatter(corrs3_data, corrs3_pair, s=1, alpha=0.05)
corrs3_diag_max = np.max((corrs3_data, corrs3_pair))
corrs3_diag_min = np.min((corrs3_data, corrs3_pair))
axs[0,0].plot([corrs3_diag_min, corrs3_diag_max], [corrs3_diag_min, corrs3_diag_max], linestyle="--", color="black")
axs[0,0].set_title("A)", loc="left", weight="bold")
axs[0,0].set_xlabel(r"$C_{ijk}^{ \; \mathrm{data}}$")
axs[0,0].set_ylabel(r"$C_{ijk}^{ \; \mathrm{pair}}$")
# axs[0,1].errorbar(concorrs3_M_data, concorrs3_M_pair, xerr=concorrs3_SD_data, yerr=concorrs3_SD_pair, linestyle="None", \
#               elinewidth=0.5, marker="o", markersize=2.5, color="black", ecolor="red")
# concorrs3_diag_max = np.max((concorrs3_M_data, concorrs3_M_pair)) + np.max((concorrs3_SD_pair, concorrs3_SD_data))
# concorrs3_diag_min = np.min((concorrs3_M_data, concorrs3_M_pair)) - np.max((concorrs3_SD_pair, concorrs3_SD_data))
axs[0,1].scatter(concorrs3_data, concorrs3_pair, s=1, alpha=0.05)
concorrs3_diag_max = np.max((concorrs3_data, concorrs3_pair))
concorrs3_diag_min = np.min((concorrs3_data, concorrs3_pair))
axs[0,1].plot([concorrs3_diag_min, concorrs3_diag_max], [concorrs3_diag_min, concorrs3_diag_max], linestyle="--", color="black")
axs[0,1].set_title("B)", loc="left", weight="bold")
axs[0,1].set_xlabel(r"$\widetilde{C}_{ijk}^{ \; \mathrm{data}}$")
axs[0,1].set_ylabel(r"$\widetilde{C}_{ijk}^{ \; \mathrm{pair}}$")

corrs3_data = corrs3[f"N={n}"]["corrs3_data"][pop2]
corrs3_pair = corrs3[f"N={n}"]["corrs3_pair"][pop2]
concorrs3_data = concorrs3[f"N={n}"]["corrs3_data"][pop2]
concorrs3_pair = concorrs3[f"N={n}"]["corrs3_pair"][pop2]

# nr_bins = 1000
# corrs3_bins_edges = np.quantile(corrs3_data, np.linspace(0, 1, nr_bins + 1, endpoint=True))
# concorrs3_bins_edges = np.quantile(concorrs3_data, np.linspace(0, 1, nr_bins + 1, endpoint=True))
# corrs3_M_data, corrs3_SD_data = np.zeros(nr_bins), np.zeros(nr_bins)
# corrs3_M_pair, corrs3_SD_pair = np.zeros(nr_bins), np.zeros(nr_bins)
# concorrs3_M_data, concorrs3_SD_data = np.zeros(nr_bins), np.zeros(nr_bins)
# concorrs3_M_pair, concorrs3_SD_pair = np.zeros(nr_bins), np.zeros(nr_bins)
# for b in range(nr_bins):
#     corrs3_mask = np.logical_and(corrs3_data > corrs3_bins_edges[b], corrs3_data < corrs3_bins_edges[b+1])
#     concorrs3_mask = np.logical_and(concorrs3_data > concorrs3_bins_edges[b], concorrs3_data < concorrs3_bins_edges[b+1])
#     corrs3_M_data[b], corrs3_SD_data[b] = np.mean(corrs3_data[corrs3_mask]), np.std(corrs3_data[corrs3_mask])
#     corrs3_M_pair[b], corrs3_SD_pair[b] = np.mean(corrs3_pair[corrs3_mask]), np.std(corrs3_pair[corrs3_mask])
#     concorrs3_M_data[b], concorrs3_SD_data[b] = np.mean(concorrs3_data[concorrs3_mask]), np.std(concorrs3_data[concorrs3_mask])
#     concorrs3_M_pair[b], concorrs3_SD_pair[b] = np.mean(concorrs3_pair[concorrs3_mask]), np.std(concorrs3_pair[concorrs3_mask])

# axs[1,0].errorbar(corrs3_M_data, corrs3_M_pair, xerr=corrs3_SD_data, yerr=corrs3_SD_pair, linestyle="None", \
#               elinewidth=0.5, marker="o", markersize=2.5, color="black", ecolor="red")
# corrs3_diag_max = np.max((corrs3_M_data, corrs3_M_pair)) + np.max((corrs3_SD_pair, corrs3_SD_data))
# corrs3_diag_min = np.min((corrs3_M_data, corrs3_M_pair)) - np.max((corrs3_SD_pair, corrs3_SD_data))
axs[1,0].scatter(corrs3_data, corrs3_pair, s=1, alpha=0.05)
corrs3_diag_max = np.max((corrs3_data, corrs3_pair))
corrs3_diag_min = np.min((corrs3_data, corrs3_pair))
axs[1,0].plot([corrs3_diag_min, corrs3_diag_max], [corrs3_diag_min, corrs3_diag_max], linestyle="--", color="black")
axs[1,0].set_title("C)", loc="left", weight="bold")
axs[1,0].set_xlabel(r"$C_{ijk}^{ \; \mathrm{data}}$")
axs[1,0].set_ylabel(r"$C_{ijk}^{ \; \mathrm{pair}}$")
# axs[1,1].errorbar(concorrs3_M_data, concorrs3_M_pair, xerr=concorrs3_SD_data, yerr=concorrs3_SD_pair, linestyle="None", \
#               elinewidth=0.5, marker="o", markersize=2.5, color="black", ecolor="red")
# concorrs3_diag_max = np.max((concorrs3_M_data, concorrs3_M_pair)) + np.max((concorrs3_SD_pair, concorrs3_SD_data))
# concorrs3_diag_min = np.min((concorrs3_M_data, concorrs3_M_pair)) - np.max((concorrs3_SD_pair, concorrs3_SD_data))
axs[1,1].scatter(concorrs3_data, concorrs3_pair, s=1, alpha=0.05)
concorrs3_diag_max = np.max((concorrs3_data, concorrs3_pair))
concorrs3_diag_min = np.min((concorrs3_data, concorrs3_pair))
axs[1,1].plot([concorrs3_diag_min, concorrs3_diag_max], [concorrs3_diag_min, concorrs3_diag_max], linestyle="--", color="black")
axs[1,1].set_title("D)", loc="left", weight="bold")
axs[1,1].set_xlabel(r"$\widetilde{C}_{ijk}^{ \; \mathrm{data}}$")
axs[1,1].set_ylabel(r"$\widetilde{C}_{ijk}^{ \; \mathrm{pair}}$")

fig.tight_layout()

# %% Comparing third-order interactions as a substitute for G

with open("Simulations/corrs3ver2_2-100neurons.pkl", "rb") as file:
    corrs3 = np.load(file, allow_pickle=True)
with open("Simulations/concorrs3_3-100neurons.pkl", "rb") as file:
    concorrs3 = np.load(file, allow_pickle=True)
print(corrs3["Info"])
print(concorrs3["Info"])

neurons = np.arange(5, 48, 1)
binsize = 0.02 
populations = 100

Gs_corrs3 = np.full((populations, len(neurons)), np.nan)
Gs_concorrs3 = np.full((populations, len(neurons)), np.nan)
perturb_params = np.full((populations, len(neurons)), np.nan) # perturb_params is the same for corrs3 and concorrs3
# firing_rates = np.full((populations, len(neurons)), np.nan)
for n_idx, n in enumerate(neurons):
    corrs3_data = corrs3[f"N={n}"]["corrs3_data"]
    corrs3_pair = corrs3[f"N={n}"]["corrs3_pair"]
    corrs3_ind = corrs3[f"N={n}"]["corrs3_ind"]
    concorrs3_data = concorrs3[f"N={n}"]["corrs3_data"]
    concorrs3_pair = concorrs3[f"N={n}"]["corrs3_pair"]
    concorrs3_ind = concorrs3[f"N={n}"]["corrs3_ind"]
    Gs_corrs3[:, n_idx] = 1 - np.sqrt(np.mean((corrs3_data - corrs3_pair)**2, axis=1)) / \
                              np.sqrt(np.mean((corrs3_data - corrs3_ind)**2, axis=1))
    Gs_concorrs3[:, n_idx] = 1 - np.sqrt(np.mean((concorrs3_data - concorrs3_pair)**2, axis=1)) / \
                                 np.sqrt(np.mean((concorrs3_data - concorrs3_ind)**2, axis=1))
    perturb_params[:, n_idx] = corrs3[f"N={n}"]["perturb_params"] # perturb_params is the same for corrs3 and concorrs3
    # firing_rates[:, n_idx] = perturb_params[:, n_idx] / (n * binsize)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
fig.suptitle(f"Performance of pairwise model up to $N={neurons[-1]}$ as measured by" + r" $G_C$ and $G_{\widetilde{C}}$")
ax1.scatter(perturb_params, Gs_corrs3, s=1, alpha=0.05)
ax1.set_title("Third-order correlations")
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel(r"$N \bar v \delta t$")
ax1.set_ylabel("$G_C$")
ax1.set_ylim(0., 1.1) 
ax1.set_xticks(np.arange(0, 16, 1), minor=True)
ax1.set_xticks(np.arange(0, 16, 5))
mask_corrs3 = np.logical_and(Gs_corrs3 > -0.02, Gs_corrs3 < 1.1) # removes outliers
b = 1
bins_edges = np.arange(0, 16, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs_corrs3[mask_corrs3][np.logical_and(perturb_params[mask_corrs3] >= bins_edges[i], perturb_params[mask_corrs3] <= bins_edges[i+1])]) 
    SD_G[i] = np.nanstd(Gs_corrs3[mask_corrs3][np.logical_and(perturb_params[mask_corrs3] >= bins_edges[i], perturb_params[mask_corrs3] <= bins_edges[i+1])])
ax1.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax2.scatter(perturb_params, Gs_concorrs3, s=1, alpha=0.05)
ax2.set_title("Connected third-order correlations")
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel(r"$N \bar v \delta t$")
ax2.set_ylabel(r"$G_{\widetilde{C}}$")
ax2.set_ylim(-1.1, 1.1)
ax2.set_xticks(np.arange(0, 16, 1), minor=True)
ax2.set_xticks(np.arange(0, 16, 5))
mask_concorrs3 = np.logical_and(Gs_concorrs3 > -0.02, Gs_concorrs3 < 1.1) # removes outliers
b = 1
bins_edges = np.arange(0, 16, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs_concorrs3[mask_concorrs3][np.logical_and(perturb_params[mask_concorrs3] >= bins_edges[i], perturb_params[mask_concorrs3] <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs_concorrs3[mask_concorrs3][np.logical_and(perturb_params[mask_concorrs3] >= bins_edges[i], perturb_params[mask_concorrs3] <= bins_edges[i+1])])
ax2.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
fig.tight_layout()

print(np.sum(Gs_corrs3 < 0), np.sum(Gs_corrs3 > 1.1)) 
print(np.sum(Gs_concorrs3 < -1.1), np.sum(Gs_concorrs3 > 1.1)) # All 509 outliers are for N < 

# %% Example of the number of simultanously active neurons in the pairwise model and the data

n, pop = 100, 0

with open("Simulations/SimulSpikes_2-100neurons.pkl", "rb") as file:
    SimulSpikes = pickle.load(file)

print(SimulSpikes["Info"])

SimulSpikes_data = SimulSpikes[f"N={n}"]["SimulSpikes_data"][pop]
SimulSpikes_pair = SimulSpikes[f"N={n}"]["SimulSpikes_pair"][pop]

SimulSpikes_data = SimulSpikes_data / np.sum(SimulSpikes_data)
SimulSpikes_pair = SimulSpikes_pair / np.sum(SimulSpikes_pair)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 0.73)))
fig.suptitle("Number of simultanously active neurons in data and pairwise model")
colour_range_end = 30
colours = np.array((list(range(0, colour_range_end)) + (len(SimulSpikes_data) - colour_range_end) * [colour_range_end - 1])) # np.arange(0, len(SimulSpikes_data))
ax1.scatter(SimulSpikes_data, SimulSpikes_pair, c=colours, cmap="plasma")
diag_max = np.nanmax((SimulSpikes_data, SimulSpikes_pair))
diag_min = np.nanmin((SimulSpikes_data, SimulSpikes_pair))
ax1.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel("$H_m$ from the data")
ax1.set_ylabel("$H_m$ from the pairwise model")
ax2.plot(np.arange(0, len(SimulSpikes_data)), SimulSpikes_data, label="data")
ax2.plot(np.arange(0, len(SimulSpikes_pair)), SimulSpikes_pair, label="pairwise model")
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel("Number of simultaneously active neurons ($m$)")
ax2.set_ylabel("$H_m$")
ax2.legend()
fig.tight_layout()

# %% Comparing the number of simultanously active neurons as a substitute for G

with open("Simulations/SimulSpikes_2-100neurons_new.pkl", "rb") as file:
    SimulSpikes = pickle.load(file)

print(SimulSpikes["Info"])

neurons = np.arange(2, 101, 1)
binsize = 0.02 
populations = 100
Ns = np.tile(neurons, populations).reshape(populations, len(neurons)) 

Gs_SimulSpikes = np.full((populations, len(neurons)), np.nan)
perturb_params = np.full((populations, len(neurons)), np.nan)
for n_idx, n in enumerate(neurons):
    SimulSpikes_data = SimulSpikes[f"N={n}"]["SimulSpikes_data"]
    SimulSpikes_pair = SimulSpikes[f"N={n}"]["SimulSpikes_pair"]
    SimulSpikes_ind = SimulSpikes[f"N={n}"]["SimulSpikes_ind"]
    # SimulSpikes_data = SimulSpikes_data / np.sum(SimulSpikes_data)
    # SimulSpikes_pair = SimulSpikes_pair / np.sum(SimulSpikes_pair)
    # SimulSpikes_ind = SimulSpikes_ind / np.sum(SimulSpikes_ind)
    Gs_SimulSpikes[:, n_idx] = 1 - np.sqrt(np.mean((SimulSpikes_data - SimulSpikes_pair)**2, axis=1)) / \
                                   np.sqrt(np.mean((SimulSpikes_data - SimulSpikes_ind)**2, axis=1))
    perturb_params[:, n_idx] = SimulSpikes[f"N={n}"]["perturb_params"]

fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"]*np.array(0.85))
fig.suptitle(f"Performance of pairwise model up to $N={neurons[-1]}$ as measured by $G_H$")
ax.scatter(perturb_params, Gs_SimulSpikes, s=1, alpha=0.1)
ax.set_xlabel(r"$N \bar v \delta t$")
ax.set_ylabel("$G_H$")
ax.set_ylim(0., 1.1) 
ax.set_xticks(np.arange(0, 16, 1), minor=True)
ax.set_xticks(np.arange(0, 16, 5))
mask_SimulSpikes = np.logical_and(Gs_SimulSpikes > -0.02, Gs_SimulSpikes < 1.1) # removes outliers
b = 1
bins_edges = np.arange(0, 16, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs_SimulSpikes[mask_SimulSpikes][np.logical_and(perturb_params[mask_SimulSpikes] >= bins_edges[i], perturb_params[mask_SimulSpikes] <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs_SimulSpikes[mask_SimulSpikes][np.logical_and(perturb_params[mask_SimulSpikes] >= bins_edges[i], perturb_params[mask_SimulSpikes] <= bins_edges[i+1])])
ax.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")

print(np.sum(Gs_SimulSpikes < 0), np.sum(Gs_SimulSpikes > 1.1)) # All outliers are for N < 5

# %% Effect of finite sampling (correction and half of the data)

with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
    params_quality = pickle.load(file)
with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_05data.pkl", "rb") as file:
    params_quality_05data = pickle.load(file)
with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_corr.pkl", "rb") as file:
    params_quality_corr = pickle.load(file)

neurons = np.arange(2, 101, 1)
binsize = 0.02        
populations = 100

Gs, Gs_05data, Gs_corr = np.full((populations, len(neurons)), np.nan), np.full((populations, len(neurons)), np.nan), np.full((populations, len(neurons)), np.nan)
perturb_params, perturb_params_05data, perturb_params_corr = np.full((populations, len(neurons)), np.nan), np.full((populations, len(neurons)), np.nan), np.full((populations, len(neurons)), np.nan)
for n_idx, n in enumerate(neurons):
    Gs[:, n_idx] = params_quality[f"N={n}"]["Gs"] if n > 15 else params_quality[f"N={n}"]["Gs_plugin"]
    perturb_params[:, n_idx] = params_quality[f"N={n}"]["perturb_params"]
    Gs_05data[:, n_idx] = params_quality_05data[f"N={n}"]["Gs"] if n > 15 else params_quality_05data[f"N={n}"]["Gs_plugin"]
    perturb_params_05data[:, n_idx] = params_quality_corr[f"N={n}"]["perturb_params"]
    Gs_corr[:, n_idx] = params_quality_corr[f"N={n}"]["Gs"] if n > 15 else params_quality_corr[f"N={n}"]["Gs_plugin"]
    perturb_params_corr[:, n_idx] = params_quality_corr[f"N={n}"]["perturb_params"]

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=plt.rcParams["figure.figsize"]*np.array((1.30, 0.64)))
fig.suptitle(f"Effect of finite sampling up to $N={neurons[-1]}$")
ax1.scatter(perturb_params, Gs, s=1, alpha=0.05)
ax1.set_title(r"Original $\hat{G}$")
ax1.set_title("A)", loc="left", weight="bold")
ax1.set_xlabel(r"$N \bar v \delta t$")
ax1.set_ylabel(r"$\hat{G}$")
ax1.set_ylim(0., 1.1) 
ax2.scatter(perturb_params_05data, Gs_05data, s=1, alpha=0.05)
ax2.set_title(r"$\hat{G}$ with half of the data")
ax2.set_title("B)", loc="left", weight="bold")
ax2.set_xlabel(r"$N \bar v \delta t$")
# ax2.set_ylabel("$G$")
ax2.set_ylim(0., 1.1)
ax3.scatter(perturb_params_corr, Gs_corr, s=1, alpha=0.05)
ax3.set_title(r"Corrected $\hat{G}$")
ax3.set_title("C)", loc="left", weight="bold")
ax3.set_xlabel(r"$N \bar v \delta t$")
# ax3.set_ylabel("$G$")
ax3.set_ylim(0., 1.1)
fig.tight_layout()

# Make M and SD lines
b = 1
bins_edges = np.arange(0, 16, b)
nr_bins = len(bins_edges) - 1
M_G, SD_G = np.zeros(nr_bins), np.zeros(nr_bins)
M_G_05data, SD_G_05data = np.zeros(nr_bins), np.zeros(nr_bins)
M_G_corr, SD_G_corr = np.zeros(nr_bins), np.zeros(nr_bins)
for i in range(nr_bins):
    M_G[i] = np.nanmean(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    SD_G[i] = np.nanstd(Gs[np.logical_and(perturb_params >= bins_edges[i], perturb_params <= bins_edges[i+1])])
    M_G_05data[i] = np.nanmean(Gs_05data[np.logical_and(perturb_params_05data >= bins_edges[i], perturb_params_05data <= bins_edges[i+1])])
    SD_G_05data[i] = np.nanstd(Gs_05data[np.logical_and(perturb_params_05data >= bins_edges[i], perturb_params_05data <= bins_edges[i+1])])
    M_G_corr[i] = np.nanmean(Gs_corr[np.logical_and(perturb_params_corr >= bins_edges[i], perturb_params_corr <= bins_edges[i+1])])
    SD_G_corr[i] = np.nanstd(Gs_corr[np.logical_and(perturb_params_corr >= bins_edges[i], perturb_params_corr <= bins_edges[i+1])])
ax1.errorbar(bins_edges[:-1] + b/2, M_G, yerr=SD_G, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax1.set_xticks(bins_edges, minor=True)
ax1.set_xticks(bins_edges[::5])
ax2.errorbar(bins_edges[:-1] + b/2, M_G_05data, yerr=SD_G_05data, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax2.set_xticks(bins_edges, minor=True)
ax2.set_xticks(bins_edges[::5])
ax3.errorbar(bins_edges[:-1] + b/2, M_G_corr, yerr=SD_G_corr, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
ax3.set_xticks(bins_edges, minor=True)
ax3.set_xticks(bins_edges[::5])

# %% Effect of finite sampling for N=20 and N=100 (correction and half of the data)

# with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100.pkl", "rb") as file:
#     params_quality_20N = pickle.load(file)
# with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_05data.pkl", "rb") as file:
#     params_quality_20N_05data = pickle.load(file)
# with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_corr.pkl", "rb") as file:
#     params_quality_20N_corr = pickle.load(file)
# with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100.pkl", "rb") as file:
#     params_quality_100N = pickle.load(file)
# with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_05data.pkl", "rb") as file:
#     params_quality_100N_05data = pickle.load(file)
# with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_corr.pkl", "rb") as file:
#     params_quality_100N_corr = pickle.load(file)
    
with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_new.pkl", "rb") as file:
    params_quality_20N = pickle.load(file)
with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_05data_new.pkl", "rb") as file:
    params_quality_20N_05data = pickle.load(file)
with open("Simulations/G_PL_binsize0.02_neurons2-20_populations100_corr_new.pkl", "rb") as file:
    params_quality_20N_corr = pickle.load(file)
with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_new.pkl", "rb") as file:
    params_quality_100N = pickle.load(file)
with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_05data_new.pkl", "rb") as file:
    params_quality_100N_05data = pickle.load(file)
with open("Simulations/Ghat_PL_binsize0.02_neurons2-100_populations100_corr_new.pkl", "rb") as file:
    params_quality_100N_corr = pickle.load(file)

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=plt.rcParams["figure.figsize"]*np.array((1.27, 1.9)))
fig.suptitle("Effect of finite sampling", y=0.99)

binsize = 0.02        
populations = 100

neurons20 = np.arange(2, 19, 1)
Gs_20N, Gs_20N_05data, Gs_20N_corr = np.full((populations, len(neurons20)), np.nan), np.full((populations, len(neurons20)), np.nan), np.full((populations, len(neurons20)), np.nan)
perturb_params_20N, perturb_params_20N_05data, perturb_params_20N_corr = np.full((populations, len(neurons20)), np.nan), np.full((populations, len(neurons20)), np.nan), np.full((populations, len(neurons20)), np.nan)
for n_idx, n in enumerate(neurons20):
    Gs_20N[:, n_idx] = params_quality_20N[f"N={n}"]["Gs"] 
    perturb_params_20N[:, n_idx] = params_quality_20N[f"N={n}"]["perturb_params"]
    Gs_20N_05data[:, n_idx] = params_quality_20N_05data[f"N={n}"]["Gs"] 
    perturb_params_20N_05data[:, n_idx] = params_quality_20N_05data[f"N={n}"]["perturb_params"]
    Gs_20N_corr[:, n_idx] = params_quality_20N_corr[f"N={n}"]["Gs"] 
    perturb_params_20N_corr[:, n_idx] = params_quality_20N_corr[f"N={n}"]["perturb_params"]

neurons100 = np.arange(2, 48, 1)
Gs_100N, Gs_100N_05data, Gs_100N_corr = np.full((populations, len(neurons100)), np.nan), np.full((populations, len(neurons100)), np.nan), np.full((populations, len(neurons100)), np.nan)
perturb_params_100N, perturb_params_100N_05data, perturb_params_100N_corr = np.full((populations, len(neurons100)), np.nan), np.full((populations, len(neurons100)), np.nan), np.full((populations, len(neurons100)), np.nan)
for n_idx, n in enumerate(neurons100):
    Gs_100N[:, n_idx] = params_quality_100N[f"N={n}"]["Gs"] if n > 15 else params_quality_100N[f"N={n}"]["Gs_plugin"]
    perturb_params_100N[:, n_idx] = params_quality_100N[f"N={n}"]["perturb_params"]
    Gs_100N_05data[:, n_idx] = params_quality_100N_05data[f"N={n}"]["Gs"] if n > 15 else params_quality_100N_05data[f"N={n}"]["Gs_plugin"]
    perturb_params_100N_05data[:, n_idx] = params_quality_100N_05data[f"N={n}"]["perturb_params"]
    Gs_100N_corr[:, n_idx] = params_quality_100N_corr[f"N={n}"]["Gs"] if n > 15 else params_quality_100N_corr[f"N={n}"]["Gs_plugin"]
    perturb_params_100N_corr[:, n_idx] = params_quality_100N_corr[f"N={n}"]["perturb_params"]

axs[0,0].scatter(perturb_params_20N, Gs_20N, s=1, alpha=0.1)
axs[0,0].set_title("Up to $N=20$ neurons")
axs[0,0].set_title("A)", loc="left", weight="bold")
# axs[0,0].set_xlabel(r"$N \bar v \delta t$")
axs[0,0].set_ylabel(r"$G$")  
axs[0,0].annotate("Original $G$", xy=(-0.23, 0.5), xycoords=axs[0,0].transAxes, rotation="vertical", va="center", fontsize=12) 
axs[0,0].set_ylim(0., 1.1) 
axs[1,0].scatter(perturb_params_20N_05data, Gs_20N_05data, s=1, alpha=0.1)
axs[1,0].set_title("C)", loc="left", weight="bold")
# axs[1,0].set_xlabel(r"$N \bar v \delta t$")
axs[1,0].set_ylabel("$G$")
axs[1,0].annotate("$G$ with half of the data", xy=(-0.23, 0.5), xycoords=axs[1,0].transAxes, rotation="vertical", va="center", fontsize=12) 
axs[1,0].set_ylim(0., 1.1)
axs[2,0].scatter(perturb_params_20N_corr, Gs_20N_corr, s=1, alpha=0.1)
axs[2,0].set_title("E)", loc="left", weight="bold")
axs[2,0].set_xlabel(r"$N \bar v \delta t$")
axs[2,0].set_ylabel("$G$")
axs[2,0].annotate("Corrected $G$", xy=(-0.23, 0.5), xycoords=axs[2,0].transAxes, rotation="vertical", va="center", fontsize=12) 
axs[2,0].set_ylim(0., 1.1)

axs[0,1].scatter(perturb_params_100N, Gs_100N, s=1, alpha=0.05)
axs[0,1].set_title("Up to $N=100$ neurons")
axs[0,1].set_title("B)", loc="left", weight="bold")
# axs[0,1].set_xlabel(r"$N \bar v \delta t$")
axs[0,1].set_ylabel(r"$\hat{G}$")
axs[0,1].set_ylim(0., 1.1) 
axs[1,1].scatter(perturb_params_100N_05data, Gs_100N_05data, s=1, alpha=0.05)
axs[1,1].set_title("E)", loc="left", weight="bold")
# axs[1,1].set_xlabel(r"$N \bar v \delta t$")
axs[1,1].set_ylabel(r"$\hat{G}$")
axs[1,1].set_ylim(0., 1.1)
axs[2,1].scatter(perturb_params_100N_corr, Gs_100N_corr, s=1, alpha=0.05)
axs[2,1].set_title("F)", loc="left", weight="bold")
axs[2,1].set_xlabel(r"$N \bar v \delta t$")
axs[2,1].set_ylabel(r"$\hat{G}$")
axs[2,1].set_ylim(0., 1.1)

# Make M and SD lines
b = 1
bins_edges_20N = np.arange(0, 5, b)
nr_bins_20N = len(bins_edges_20N) - 1
M_G_20N, SD_G_20N = np.zeros(nr_bins_20N), np.zeros(nr_bins_20N)
M_G_20N_05data, SD_G_20N_05data = np.zeros(nr_bins_20N), np.zeros(nr_bins_20N)
M_G_20N_corr, SD_G_20N_corr = np.zeros(nr_bins_20N), np.zeros(nr_bins_20N)
for i in range(nr_bins_20N):
    M_G_20N[i] = np.nanmean(Gs_20N[np.logical_and(perturb_params_20N >= bins_edges_20N[i], perturb_params_20N <= bins_edges_20N[i+1])])
    SD_G_20N[i] = np.nanstd(Gs_20N[np.logical_and(perturb_params_20N >= bins_edges_20N[i], perturb_params_20N <= bins_edges_20N[i+1])])
    M_G_20N_05data[i] = np.nanmean(Gs_20N_05data[np.logical_and(perturb_params_20N_05data >= bins_edges_20N[i], perturb_params_20N_05data <= bins_edges_20N[i+1])])
    SD_G_20N_05data[i] = np.nanstd(Gs_20N_05data[np.logical_and(perturb_params_20N_05data >= bins_edges_20N[i], perturb_params_20N_05data <= bins_edges_20N[i+1])])
    M_G_20N_corr[i] = np.nanmean(Gs_20N_corr[np.logical_and(perturb_params_20N_corr >= bins_edges_20N[i], perturb_params_20N_corr <= bins_edges_20N[i+1])])
    SD_G_20N_corr[i] = np.nanstd(Gs_20N_corr[np.logical_and(perturb_params_20N_corr >= bins_edges_20N[i], perturb_params_20N_corr <= bins_edges_20N[i+1])])
axs[0,0].errorbar(bins_edges_20N[:-1] + b/2, M_G_20N, yerr=SD_G_20N, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[0,0].set_xticks(bins_edges_20N)
axs[0,0].tick_params(labelbottom=False) 
axs[1,0].errorbar(bins_edges_20N[:-1] + b/2, M_G_20N_05data, yerr=SD_G_20N_05data, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[1,0].set_xticks(bins_edges_20N)
axs[1,0].tick_params(labelbottom=False) 
axs[2,0].errorbar(bins_edges_20N[:-1] + b/2, M_G_20N_corr, yerr=SD_G_20N_corr, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[2,0].set_xticks(bins_edges_20N)

b = 1
bins_edges_100N = np.arange(0, 16, b)
nr_bins_100N = len(bins_edges_100N) - 1
M_G_100N, SD_G_100N = np.zeros(nr_bins_100N), np.zeros(nr_bins_100N)
M_G_100N_05data, SD_G_100N_05data = np.zeros(nr_bins_100N), np.zeros(nr_bins_100N)
M_G_100N_corr, SD_G_100N_corr = np.zeros(nr_bins_100N), np.zeros(nr_bins_100N)
for i in range(nr_bins_100N):
    M_G_100N[i] = np.nanmean(Gs_100N[np.logical_and(perturb_params_100N >= bins_edges_100N[i], perturb_params_100N <= bins_edges_100N[i+1])])
    SD_G_100N[i] = np.nanstd(Gs_100N[np.logical_and(perturb_params_100N >= bins_edges_100N[i], perturb_params_100N <= bins_edges_100N[i+1])])
    M_G_100N_05data[i] = np.nanmean(Gs_100N_05data[np.logical_and(perturb_params_100N_05data >= bins_edges_100N[i], perturb_params_100N_05data <= bins_edges_100N[i+1])])
    SD_G_100N_05data[i] = np.nanstd(Gs_100N_05data[np.logical_and(perturb_params_100N_05data >= bins_edges_100N[i], perturb_params_100N_05data <= bins_edges_100N[i+1])])
    M_G_100N_corr[i] = np.nanmean(Gs_100N_corr[np.logical_and(perturb_params_100N_corr >= bins_edges_100N[i], perturb_params_100N_corr <= bins_edges_100N[i+1])])
    SD_G_100N_corr[i] = np.nanstd(Gs_100N_corr[np.logical_and(perturb_params_100N_corr >= bins_edges_100N[i], perturb_params_100N_corr <= bins_edges_100N[i+1])])
axs[0,1].errorbar(bins_edges_100N[:-1] + b/2, M_G_100N, yerr=SD_G_100N, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[0,1].set_xticks(bins_edges_100N, minor=True)
axs[0,1].set_xticks(bins_edges_100N[::5])
axs[0,1].tick_params(labelbottom=False) 
axs[1,1].errorbar(bins_edges_100N[:-1] + b/2, M_G_100N_05data, yerr=SD_G_100N_05data, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[1,1].set_xticks(bins_edges_100N, minor=True)
axs[1,1].set_xticks(bins_edges_100N[::5])
axs[1,1].tick_params(labelbottom=False) 
axs[2,1].errorbar(bins_edges_100N[:-1] + b/2, M_G_100N_corr, yerr=SD_G_100N_corr, linestyle="-", elinewidth=1.0, \
            marker="o", markersize=2.5, color="black", ecolor="black")
axs[2,1].set_xticks(bins_edges_100N, minor=True)
axs[2,1].set_xticks(bins_edges_100N[::5])

fig.tight_layout()
