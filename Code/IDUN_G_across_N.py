import numpy as np
import pickle
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

run = int(sys.argv[1]) - 1 
# eta = 0.01
# samples = 5000
# iterations = 60000
populations = 100 
binsize = 0.02
neurons = np.arange(2, 21, 1) 
rng = np.random.default_rng(42 + run) # Seed?

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

# KL_ind, KL_pair = np.empty(populations), np.empty(populations)
h, J = np.empty((populations, neurons[run])), np.empty((populations, neurons[run], neurons[run]))
# hs = np.empty((populations, iterations + 1, neurons[run])) # only for Boltzmann learning
# Js = np.empty((populations, iterations + 1, neurons[run], neurons[run])) # only for Boltzmann learning
neuron_indices = np.empty((populations, neurons[run]), dtype=int) 
perturb_params = np.empty(populations)
Gs = np.empty(populations) 
for p in range(populations):
    neuron_indices[p] = rng.choice(len(spike_times), neurons[run], replace=False) 
    subpop_spike_trains = time_binning(spike_times[neuron_indices[p]], interval, binsize)
    
    # # Use only half of the samples/data 
    subpop_spike_trains = subpop_spike_trains[rng.choice(int(subpop_spike_trains.shape[0]), int(subpop_spike_trains.shape[0] * 0.5), replace=False)]
    
    # h_initial, J_initial = np.zeros(neurons[run]), np.zeros((neurons[run], neurons[run])) 
    # h[p], J[p], hs[p], Js[p] = Boltzmann_learning(spikes_n, h_initial, J_initial, eta, iterations, samples, rng)
    h[p], J[p] = pseudolikelihood_sklearn(subpop_spike_trains) # lots faster
    
    mean_rate = np.count_nonzero(subpop_spike_trains + 1, axis=0) / (subpop_spike_trains.shape[0] * binsize)
    perturb_params[p] = neurons[run] * np.mean(mean_rate) * binsize # Equivalently: np.mean(np.count_nonzero(spikes_n + 1, axis=1))
    
    Gs[p], _, _ = G_plugin(subpop_spike_trains, h[p], J[p])
    # rng_correction = np.random.default_rng(run) 
    # Gs[p], _, _ = finite_samples_correction(subpop_spike_trains, G_plugin, (None, None), rng_correction)

filename = f"G_PL_binsize{binsize}_neurons{neurons[0]}-{neurons[-1]}_populations{populations}_05data_new.pkl"
try:
    with open(filename, "rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = pickle.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)
    params_quality[f"N={neurons[run]}"] = \
        {"h":h, "J":J, "neuron_indices":neuron_indices , "Gs":Gs, "perturb_params":perturb_params}
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"used concatenated spike times from concatenated_spike_times_26471_bank0_495neurons_6sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             only half of the data was used for each population; \
                             a seed of 42 + run was used for reproducibility"
    params_quality[f"N={neurons[run]}"] = \
        {"h":h, "J":J, "neuron_indices":neuron_indices , "Gs":Gs, "perturb_params":perturb_params}
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)