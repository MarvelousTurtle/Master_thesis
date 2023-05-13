import numpy as np
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

population_nr = int(sys.argv[1]) - 1 
N = 20
b = 0.02
timesteps = 400000
nr_populations = 100
rng = np.random.default_rng(N + population_nr)

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

subpop = rng.choice(len(spike_times), N, replace=False)
spike_trains_subpop = time_binning(spike_times[subpop], interval, b)
spike_trains_subpop = spike_trains_subpop[rng.choice(spike_trains_subpop.shape[0], timesteps, replace=False)]

# SD_params = 1/np.sqrt(N-1) 
# h = rng.normal(0, SD_params, N) 
# J = np.zeros((N, N))
# J[np.triu_indices(N, 1)] = rng.normal(0, SD_params, int(N*(N-1)/2))
# J = (J + J.T) 
# J = np.tril(J, k=-1) 

# spike_trains_subpop = Metropolis_samples(timesteps, h, J, 0.5, rng)

h, J = pseudolikelihood_sklearn(spike_trains_subpop) # if we pick samples in which one neuron never fires pseudolikelihood_sklearn() will not work
# eta = 0.001
# nr_samples = 5000
# nr_iterations = 40000
# h_initial, J_initial = np.zeros(N), np.zeros((N, N)) 
# h, J, _, _ = Boltzmann_learning(spike_trains_subpop, h_initial, J_initial, eta, nr_iterations, nr_samples)

G, _, _ = G_plugin_RC(spike_trains_subpop, h, J)
G_approx, _, _ = Ghat_RC(spike_trains_subpop, h, J)

filename = f"Ghat_G_NeuralData2_{N}neurons_{nr_populations}populations_timesteps{timesteps}.npz" 
try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        quality = np.load(file, allow_pickle=True)
        Gs_file, Gs_approx_file = quality["Gs"], quality["Gs_approx"]
        Gs_file[population_nr], Gs_approx_file[population_nr]  = G, G_approx
        np.savez(file, info=quality["info"], Gs=Gs_file, Gs_approx=Gs_approx_file, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    info = f"used concatenated spike times from concatenated_spike_times_SomCor_26504_bank1_287neurons_4sessions.npz; \
             Comparing uncorrected Ghat and G for {nr_populations} populations of synthetic data. \
             Pseudolikelihood was used to approximate the parameters. \
             neurons = {N}, timesteps = {timesteps}, seed = N + nr_populations" 
    Gs, Gs_approx  = np.zeros(nr_populations), np.zeros(nr_populations)
    Gs[population_nr], Gs_approx[population_nr]  = G, G_approx
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, info=info, Gs=Gs, Gs_approx=Gs_approx, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)

