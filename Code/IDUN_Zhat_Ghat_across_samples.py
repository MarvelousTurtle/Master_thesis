import numpy as np
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

run = int(sys.argv[1]) - 1
populations = 200
rng = np.random.default_rng(run)

N = 15
samples_start, samples_end = 100, 20100
spacing = 400

# binsize = 0.02
# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
#     data = np.load(file, allow_pickle=True)
#     spike_times, interval = data["spike_times"], data["interval"]
# data = time_binning(spike_times[rng.choice(len(spike_times), N, replace=False)], interval, binsize)
# data = data[rng.choice(data.shape[0], samples_end, replace=False)]

SD_params = 1/np.sqrt(N-1) 
h_gen = rng.normal(0, SD_params, N) 
J_gen = np.zeros((N, N))
J_gen[np.triu_indices(N, 1)] = rng.normal(0, SD_params, int(N*(N-1)/2))
J_gen = (J_gen + J_gen.T) 
J_gen = np.tril(J_gen, k=-1) 
data = Metropolis_samples(samples_end + 5000, h_gen, J_gen, 0.5, rng)[5000:]

Z = 0
for s in itertools.product([-1, 1], repeat=len(h_gen)):
    s = np.array(s)
    Z += np.exp(np.sum(J_gen * s[None, :] * s[:, None]) + np.sum(h_gen * s))

eta = 0.001
nr_samples = 5000
nr_iterations = 30000
h_initial , J_initial = np.zeros(N), np.zeros((N, N)) 

data_props = np.arange(samples_start, samples_end + 1, spacing)
Zs_mean = np.zeros(len(data_props))
Zs_median = np.zeros(len(data_props))
Zs_MostSampled = np.zeros(len(data_props))
Zs_hat = np.zeros(len(data_props))
Zs = np.zeros(len(data_props))
Gs = np.zeros(len(data_props))
Gs_hat = np.zeros(len(data_props))
for idx, p in enumerate(data_props):
    h, J, _, _, = Boltzmann_learning(data[0:int(p)], h_initial, J_initial, eta, nr_iterations, nr_samples, rng)
    # h, J = pseudolikelihood_sklearn(data[0:int(p)]) 
    J = np.tril((J + J.T) / 2, k=-1) # !!
    
    for s in itertools.product([-1, 1], repeat=len(h)):
        s = np.array(s)
        Zs[idx] += np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
    
    sampled_states_data, counts = np.unique(data[0:int(p)], axis=0, return_counts=True)
    dist_data = counts / data[0:int(p)].shape[0]
    h_ind = np.arctanh(np.mean(data[0:int(p)], axis=0))
    e_pair = np.empty(len(sampled_states_data))
    dist_ind = np.empty(len(sampled_states_data))
    for idx_s, s in enumerate(sampled_states_data):
        e_pair[idx_s] = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        dist_ind[idx_s] = np.prod(np.exp(s * h_ind) / (np.exp(h_ind) + np.exp(-h_ind)))
    Zs_mean[idx] = np.mean(e_pair / dist_data)
    Zs_median[idx] = np.median(e_pair / dist_data)
    Zs_MostSampled[idx] = e_pair[np.argmax(dist_data)] / np.max(dist_data)
    Zs_hat[idx] = np.sum(e_pair * e_pair) / np.sum(dist_data * e_pair) 
    KL_ind = np.sum(dist_data * np.log2(dist_data / dist_ind))
    KL_pair_hat = np.sum(dist_data * np.log2(dist_data)) - np.sum(dist_data * np.log2(e_pair)) + np.log2(Zs_hat[idx])
    KL_pair = np.sum(dist_data * np.log2(dist_data)) - np.sum(dist_data * np.log2(e_pair)) + np.log2(Zs[idx])
    Gs_hat[idx] = 1 - KL_pair_hat / KL_ind
    Gs[idx] = 1 - KL_pair / KL_ind
    
filename = f"Zhat_Ghat_samples{samples_start}-{samples_end}_neurons{N}_populations{populations}.npz" 

try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        Zhat_Ghat = np.load(file, allow_pickle=True)
        Z_f = Zhat_Ghat["Z"]
        Zs_f, Zs_hat_f = Zhat_Ghat["Zs"], Zhat_Ghat["Zs_hat"]
        Zs_mean_f, Zs_median_f, Zs_MostSampled_f = Zhat_Ghat["Zs_mean"], Zhat_Ghat["Zs_median"], Zhat_Ghat["Zs_MostSampled"]
        Gs_f, Gs_hat_f = Zhat_Ghat["Gs"], Zhat_Ghat["Gs_hat"]
        Z_f[run] = Z
        Zs_f[run], Zs_hat_f[run] = Zs, Zs_hat
        Zs_mean_f[run], Zs_median_f[run], Zs_MostSampled_f[run] = Zs_mean, Zs_median, Zs_MostSampled
        Gs_f[run], Gs_hat_f[run] = Gs, Gs_hat
        np.savez(file, info=Zhat_Ghat["info"], Z=Z_f, Zs=Zs_f, Zs_hat=Zs_hat_f, Zs_mean=Zs_mean_f, \
                 Zs_median=Zs_median_f, Zs_MostSampled=Zs_MostSampled_f, Gs=Gs_f, Gs_hat=Gs_hat_f, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    Zhat_Ghat = {}
    Zhat_Ghat["info"] = f"Approximations of Z and G for np.arange({samples_start}, {samples_end} + 1, {spacing}) samples. \
                        For each point, Boltzmann learning was used to approximate the model parameters and Zs from data. \
                        Boltzmann learning parameters: eta={eta}, nr_samples={nr_samples}, nr_iterations={nr_iterations}. \
                        N={N}, seed=(42 + run), nr_populations={populations}."
    Zhat_Ghat["Z"] = np.full(populations, np.nan)
    Zhat_Ghat["Z"][run] = Z
    Zhat_Ghat["Zs"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Zs"][run] = Zs
    Zhat_Ghat["Zs_hat"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Zs_hat"][run] = Zs_hat
    Zhat_Ghat["Zs_mean"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Zs_mean"][run] = Zs_mean
    Zhat_Ghat["Zs_median"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Zs_median"][run] = Zs_median
    Zhat_Ghat["Zs_MostSampled"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Zs_MostSampled"][run] = Zs_MostSampled
    Zhat_Ghat["Gs"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Gs"][run] = Gs
    Zhat_Ghat["Gs_hat"] = np.full((populations, len(data_props)), np.nan)
    Zhat_Ghat["Gs_hat"][run] = Gs_hat
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **Zhat_Ghat, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
