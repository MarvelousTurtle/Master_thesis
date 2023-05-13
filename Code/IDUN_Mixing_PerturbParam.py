import numpy as np
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]
    
population_nr = int(sys.argv[1]) - 1 
rng = np.random.default_rng(population_nr)

nr_pops = 5000
neurons = [2, 101]
binsizes = [0.005, 0.05]

n = rng.integers(neurons[0], neurons[1], 1)[0]
neuron_indices = subpopulation_spike_times(spike_times, n, interval, rng)
b = rng.uniform(binsizes[0], binsizes[1], 1)[0]
subpop_spike_trains = time_binning(spike_times[neuron_indices], interval, b)
del spike_times # saving memory

mean_rate = np.count_nonzero(subpop_spike_trains + 1, axis=0) / (subpop_spike_trains.shape[0] * b)
perturb_param = n * np.mean(mean_rate) * b # Equivalently: np.mean(np.count_nonzero(spikes_n + 1, axis=1))

h, J = pseudolikelihood_sklearn(subpop_spike_trains)
G, _, _ = Ghat(subpop_spike_trains, h, J)

filename = f"Ghat_PL_RandPopulations{nr_pops}.npz"               
try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = np.load(file, allow_pickle=True)
        Gs_file, perturb_params_file, neuron_indices_file, N_vbar_deltat_file = params_quality["Gs"], params_quality["perturb_params"], params_quality["neuron_indices"], params_quality["N_vbar_deltat"]
        Gs_file[population_nr], perturb_params_file[population_nr], neuron_indices_file[population_nr], N_vbar_deltat_file[population_nr] = G, perturb_param, neuron_indices, (n, np.mean(mean_rate), b)
        np.savez(file, Info=params_quality["Info"], Gs=Gs_file, perturb_params=perturb_params_file, neuron_indices=neuron_indices_file, N_vbar_deltat=N_vbar_deltat_file, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"used concatenated spike times from concatenated_spike_times_26471_bank0_495neurons_6sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             A random N between {neurons[0]} and {neurons[-1]} and binsize between {binsizes[0]} and {binsizes[-1]} \
                             were chosen for {nr_pops} populations, which were sampled using subpopulation_spike_times(); \
                             G was calculated with Ghat; \
                             ; G was not corrected using finite_samples_correction with G_plugin(); \
                             seed = population_nr"
    params_quality["Gs"] = np.full(nr_pops, np.nan)
    params_quality["Gs"][population_nr] = G
    params_quality["perturb_params"] = np.full(nr_pops, np.nan)
    params_quality["perturb_params"][population_nr] = perturb_param
    params_quality["neuron_indices"] = np.full(nr_pops, np.nan, dtype=object)
    params_quality["neuron_indices"][population_nr] = neuron_indices
    params_quality["N_vbar_deltat"] = np.full(nr_pops, np.nan, dtype=object)
    params_quality["N_vbar_deltat"][population_nr] = (n, np.mean(mean_rate), b)
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **params_quality, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)