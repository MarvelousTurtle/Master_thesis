import numpy as np
import pickle
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

population_nr_max = 5000
population_nr = int(sys.argv[1]) - 1 
rng = np.random.default_rng(1 + population_nr) # Seed? 

# eta = 0.01
# samples = 5000
# iterations = 60000
binsizes = [0.02] # [0.005, 0.01, 0.02] 
N = 10 

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

neuron_indices = subpopulation_spike_times(spike_times, N, interval, rng)
spike_times = spike_times[neuron_indices] # overwriting to save space

h = np.full((len(binsizes), N), np.nan)
J = np.full((len(binsizes), N, N), np.nan)
G = np.full(len(binsizes), np.nan)
perturb_param = np.full(len(binsizes), np.nan)
for b_idx, b in enumerate(binsizes):
    spikes_trains = time_binning(spike_times, interval, b)

    # h_initial, J_initial = np.zeros(N), np.zeros((N, N)) 
    # h[b_idx], J[b_idx], _, _ = Boltzmann_learning(spikes_trains, h_initial, J_initial, eta, iterations, samples, rng)
    h[b_idx], J[b_idx] = pseudolikelihood_sklearn(spikes_trains) # lots faster

    mean_rate = np.count_nonzero(spikes_trains + 1, axis=0) / (spikes_trains.shape[0] * b)
    perturb_param[b_idx] = spikes_trains.shape[1] * np.mean(mean_rate) * b # Equivalently: np.mean(np.count_nonzero(spike_trains + 1, axis=1))

    # G[b_idx], _, _ = finite_samples_correction("G_sum", spikes_trains, h[b_idx], J[b_idx])
    G[b_idx], _, _ = G_plugin(spikes_trains, h[b_idx], J[b_idx])

# Can this be made prettier?
filename = f"G_PL_RandFiringRate_neurons_{N}populations{population_nr_max}_new.npz"               
try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = np.load(file, allow_pickle=True)
        h_file, J_file, Gs_file = params_quality["h"], params_quality["J"], params_quality["Gs"]
        perturb_params_file, neuron_indices_file = params_quality["perturb_params"], params_quality["neuron_indices"]
        h_file[population_nr], J_file[population_nr], Gs_file[population_nr] = h, J, G
        perturb_params_file[population_nr], neuron_indices_file[population_nr] = perturb_param, neuron_indices
        np.savez(file, Info=params_quality["Info"], h=h_file, J=J_file, Gs=Gs_file, perturb_params=perturb_params_file, neuron_indices=neuron_indices_file, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"used concatenated spike times from concatenated_spike_times_26471_bank0_495neurons_6sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             {population_nr_max} populations of {N} neurons were sampled using subpopulation_spike_times(), and G \
                             was calculated for binsizes {binsizes}; \
                             G was not corrected using finite_samples_correction with G_plugin(); \
                             when choosing subpopulations, the exponent is 3 instead of 2; \
                             a seed of 1 + population_nr was used for reproducibility"
                             # for Boltzmann learning, a learning rate of {eta} was used with {iterations} iterations and {samples} samples per iteration
    params_quality["h"] = np.full((population_nr_max, len(binsizes), N), np.nan) 
    params_quality["h"][population_nr] = h
    params_quality["J"] = np.full((population_nr_max, len(binsizes), N, N), np.nan)
    params_quality["J"][population_nr] = J
    params_quality["Gs"] = np.full((population_nr_max, len(binsizes)), np.nan)
    params_quality["Gs"][population_nr] = G
    params_quality["perturb_params"] = np.full((population_nr_max, len(binsizes)), np.nan)
    params_quality["perturb_params"][population_nr] = perturb_param
    params_quality["neuron_indices"] = np.full((population_nr_max, N), np.nan)
    params_quality["neuron_indices"][population_nr] = neuron_indices
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **params_quality, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)

