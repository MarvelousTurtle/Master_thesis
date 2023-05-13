import numpy as np
import pickle
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys


run = int(sys.argv[1]) - 1
populations = 100 # number of populations with the same number of neurons
neurons = np.arange(2, 101, 1) # Iterable
binsize = 0.02
rng = np.random.default_rng(42 + run)

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

h, J = np.empty((populations, neurons[run])), np.empty((populations, neurons[run], neurons[run]))
neuron_indices = np.empty((populations, neurons[run]), dtype=int) 
perturb_params = np.empty(populations)
SimulSpikes_data = np.empty((populations, neurons[run] + 1))
SimulSpikes_pair = np.empty((populations, neurons[run] + 1))
SimulSpikes_ind = np.empty((populations, neurons[run] + 1))
for p in range(populations):
    neuron_indices[p] = rng.choice(len(spike_times), neurons[run], replace=False) 
    subpop_spike_trains = time_binning(spike_times[neuron_indices[p]], interval, binsize)
    
    # # Use only half of the samples/data 
    # subpop_spike_trains = subpop_spike_trains[rng.choice(int(subpop_spike_trains.shape[0]), int(subpop_spike_trains.shape[0] * 0.5), replace=False)]
    
    h[p], J[p] = pseudolikelihood_sklearn(subpop_spike_trains) 
    h_ind = np.arctanh(np.mean(subpop_spike_trains, axis=0))
    
    mean_rate = np.count_nonzero(subpop_spike_trains + 1, axis=0) / (subpop_spike_trains.shape[0] * binsize)
    perturb_params[p] = neurons[run] * np.mean(mean_rate) * binsize # Equivalently: np.mean(np.count_nonzero(spikes_n + 1, axis=1))
    
    nr_samples = subpop_spike_trains.shape[0]
    samples_pair = Metropolis_samples(nr_samples, h[p], J[p], 0.5, rng)
    samples_ind = Metropolis_samples(nr_samples, h_ind, J[p] * 0, 0.5, rng)
    SimulSpikes_data[p] = SimulSpikes(subpop_spike_trains)
    SimulSpikes_pair[p] = SimulSpikes(samples_pair)
    SimulSpikes_ind[p] = SimulSpikes(samples_ind)

filename = f"SimulSpikes_{neurons[0]}-{neurons[-1]}neurons_new.pkl"
try:
    with open(filename, "rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = pickle.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)
    params_quality[f"N={neurons[run]}"] = {"h":h, "J":J , "neuron_indices":neuron_indices , "perturb_params":perturb_params, \
                                           "SimulSpikes_data":SimulSpikes_data, "SimulSpikes_pair":SimulSpikes_pair, "SimulSpikes_ind":SimulSpikes_ind}
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["Info"] = f"Used concatenated spike times from concatenated_spike_times_26471_bank0_495neurons_6sessions.npz; \
                             pseudolikelihood_sklearn() was used to approximate h and J; \
                             Number of simultanously active neurons from data and from the pairwise model (based on as many samples as in the data)\
                             for {populations} populations of between {neurons[0]} and {neurons[-1]} neurons. \
                             Binsize of {binsize} and seed of 42 + run. "
    params_quality[f"N={neurons[run]}"] = {"h":h, "J":J , "neuron_indices":neuron_indices , "perturb_params":perturb_params, \
                                           "SimulSpikes_data":SimulSpikes_data, "SimulSpikes_pair":SimulSpikes_pair, "SimulSpikes_ind":SimulSpikes_ind}
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        pickle.dump(params_quality, file, protocol=pickle.HIGHEST_PROTOCOL)
        fcntl.flock(file, fcntl.LOCK_UN)
