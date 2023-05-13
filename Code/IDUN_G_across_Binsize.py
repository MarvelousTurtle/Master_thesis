import numpy as np
import pickle
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

nr_pops = 1000
population_nr = int(sys.argv[1]) - 1 
rng = np.random.default_rng(1 + population_nr) # Seed? 

binsize_range = [0.0005, 0.02] 
N = 10 

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

neuron_indices = rng.choice(len(spike_times), N, replace=False) 
b = rng.uniform(binsize_range[0], binsize_range[1], 1)
subpop_spike_trains = time_binning(spike_times[neuron_indices], interval, b)
del spike_times # saving memory

mean_rate = np.count_nonzero(subpop_spike_trains + 1, axis=0) / (subpop_spike_trains.shape[0] * b)
perturb_param = N * np.mean(mean_rate) * b # Equivalently: np.mean(np.count_nonzero(spikes_n + 1, axis=1))

h, J = pseudolikelihood_sklearn(subpop_spike_trains)
G, _, _ = G_plugin(subpop_spike_trains, h, J)
# G, _, _ = finite_samples_correction("G_sum", spikes_trains, h, J)

filename = f"G_PL_RandBinsize2_neurons{N}_populations{nr_pops}_new.npz"               
try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = np.load(file, allow_pickle=True)
        binsizes_file, h_file, J_file, Gs_file = params_quality["binsizes"], params_quality["h"], params_quality["J"], params_quality["Gs"]
        perturb_params_file, neuron_indices_file = params_quality["perturb_params"], params_quality["neuron_indices"]
        binsizes_file[population_nr], h_file[population_nr], J_file[population_nr], Gs_file[population_nr] = b, h, J, G
        perturb_params_file[population_nr], neuron_indices_file[population_nr] = perturb_param, neuron_indices
        np.savez(file, Info=params_quality["Info"], binsizes=binsizes_file, h=h_file, J=J_file, Gs=Gs_file, \
                 perturb_params=perturb_params_file, neuron_indices=neuron_indices_file, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"used concatenated spike times from concatenated_spike_times_26471_bank0_495neurons_6sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             {nr_pops} random {N}-neuron populations were chosen and binned with a uniformly random binsize in {binsize_range}; \
                             G was not corrected using finite_samples_correction with G_plugin(); \
                             A seed of 1 + population_nr was used for reproducibility"
    params_quality["binsizes"] = np.full(nr_pops, np.nan)
    params_quality["binsizes"][population_nr] = b
    params_quality["h"] = np.full((nr_pops, N), np.nan) 
    params_quality["h"][population_nr] = h
    params_quality["J"] = np.full((nr_pops, N, N), np.nan)
    params_quality["J"][population_nr] = J
    params_quality["Gs"] = np.full(nr_pops, np.nan)
    params_quality["Gs"][population_nr] = G
    params_quality["perturb_params"] = np.full(nr_pops, np.nan)
    params_quality["perturb_params"][population_nr] = perturb_param
    params_quality["neuron_indices"] = np.full((nr_pops, N), np.nan)
    params_quality["neuron_indices"][population_nr] = neuron_indices
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **params_quality, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)

