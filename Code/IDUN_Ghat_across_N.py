import numpy as np
import pickle
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

run = int(sys.argv[1]) - 1 
populations = 100
binsize = 0.02
neurons = np.arange(2, 101, 1) 
rng = np.random.default_rng(42 + run) # Seed?

# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_VisCor_26525_bank0_539neurons_4sessions.npz", "rb") as file: 
# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_SomCor_26504_bank1_287neurons_4sessions.npz", "rb") as file: 
with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_MotCor_26472_bank0_1115neurons_4sessions.npz", "rb") as file: 
# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_AudCor_26525_bank0_376neurons_4sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

h, J = np.empty((populations, neurons[run])), np.empty((populations, neurons[run], neurons[run]))
neuron_indices = np.empty((populations, neurons[run]), dtype=int) 
perturb_params = np.empty(populations)
Gs = np.empty(populations) 
error_ratios = np.empty(populations) 
small_N = 16 # Arbitrary small N
if neurons[run] < small_N: Gs_plugin = np.empty(populations) 
for p in range(populations):
    neuron_indices[p] = rng.choice(len(spike_times), neurons[run], replace=False) 
    subpop_spike_trains = time_binning(spike_times[neuron_indices[p]], interval, binsize)
    
    # Use only half of the samples/data 
    # subpop_spike_trains = subpop_spike_trains[rng.choice(int(subpop_spike_trains.shape[0]), int(subpop_spike_trains.shape[0] * 0.5), replace=False)]
    
    h[p], J[p] = pseudolikelihood_sklearn(subpop_spike_trains) 
    
    mean_rate = np.count_nonzero(subpop_spike_trains + 1, axis=0) / (subpop_spike_trains.shape[0] * binsize)
    perturb_params[p] = neurons[run] * np.mean(mean_rate) * binsize # Equivalently: np.mean(np.count_nonzero(spikes_n + 1, axis=1))
    
    Gs[p], _, _, error_ratios[p] = Ghat(subpop_spike_trains, h[p], J[p], True)
    if neurons[run] < small_N: Gs_plugin[p] = G_plugin(subpop_spike_trains, h[p], J[p])[0]
    # rng_correction = np.random.default_rng(run) 
    # Gs[p], _, _, error_ratios[p] = finite_samples_correction(subpop_spike_trains, Ghat, (None, None, True), rng_correction)
    # if neurons[run] < small_N: Gs_plugin[p] = finite_samples_correction(subpop_spike_trains, G_plugin, rng=rng_correction)[0]

filename = f"Ghat_PL_MotCor_binsize{binsize}_neurons{neurons[0]}-{neurons[-1]}_populations{populations}_new.pkl"
try:
    with open(filename, "rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = pickle.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)
    params_quality[f"N={neurons[run]}"] = \
        {"h":h, "J":J , "neuron_indices":neuron_indices , "Gs":Gs, "perturb_params":perturb_params, "error_ratios":error_ratios}
    if neurons[run] < small_N: params_quality[f"N={neurons[run]}"]["Gs_plugin"] = Gs_plugin
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"used concatenated spike times from /concatenated_spike_times_MotCor_26472_bank0_1115neurons_4sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             G based on the exact Z is included for N < {small_N}; \
                             a seed of 42 + run was used for reproducibility"
    params_quality[f"N={neurons[run]}"] = \
        {"h":h, "J":J , "neuron_indices":neuron_indices , "Gs":Gs, "perturb_params":perturb_params, "error_ratios":error_ratios}
    if neurons[run] < small_N: params_quality[f"N={neurons[run]}"]["Gs_plugin"] = Gs_plugin
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)
