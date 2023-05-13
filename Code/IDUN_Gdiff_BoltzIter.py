import numpy as np
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

run = int(sys.argv[1]) - 1 

N = 20
Timesteps = [100, 400000]
pops_per_timesteps = 5
pop, timesteps_idx = run % pops_per_timesteps, run // pops_per_timesteps
timesteps = Timesteps[timesteps_idx]
rng = np.random.default_rng(timesteps + pop)

SD_params = 1/np.sqrt(N-1) 
h = rng.normal(0, SD_params, N) 
J = np.zeros((N, N))
J[np.triu_indices(N, 1)] = rng.normal(0, SD_params, int(N*(N-1)/2))
J = (J + J.T) 
J = np.tril(J, k=-1) 

means = np.ones(N)
while np.any(means == 1) or np.any(means == -1): # ensuring that we don't get a divide-by-zero error in h_ind=np.arctanh(). 
    data = Metropolis_samples(timesteps, h, J, 0.5, rng)
    means = np.mean(data, axis=0)

# binsize = 0.02
# with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
#     data = np.load(file, allow_pickle=True)
#     spike_times, interval = data["spike_times"], data["interval"]
# subpop = rng.choice(len(spike_times), N, replace=False)
# data = time_binning(spike_times[subpop], interval, binsize)
# data = data[rng.choice(data.shape[0], timesteps, replace=False)]

eta = 0.001
nr_samples = 5000
nr_iterations = 40000
h_initial, J_initial = np.zeros(N), np.zeros((N, N)) 
h, J, hs, Js = Boltzmann_learning(data, h_initial, J_initial, eta, nr_iterations, nr_samples)
M_h_boltz, M_J_boltz = np.mean(hs, axis=1), np.mean(Js, axis=(1, 2))
SD_h_boltz, SD_J_boltz = np.std(hs, axis=1), np.std(Js, axis=(1, 2))

# Use G_ratios or G_diffs?
nr_evals = 100
chosen_iters = np.rint(np.linspace(0, nr_iterations - 1, nr_evals)).astype("int64")
M_h_boltz, M_J_boltz = M_h_boltz[chosen_iters], M_J_boltz[chosen_iters]
SD_h_boltz, SD_J_boltz = SD_h_boltz[chosen_iters], SD_J_boltz[chosen_iters]
G_diffs = np.zeros(nr_evals)
for idx, ite in enumerate(chosen_iters):
    G_approx, G = Ghat(data, hs[ite], Js[ite])[0], G_plugin(data, hs[ite], Js[ite])[0]
    G_diffs[idx] = G_approx - G    

filename = f"Gdiff_BoltzIter_neuorns{N}_timesteps{Timesteps[0]}-{Timesteps[-1]}_new.npz"
try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        params_quality = np.load(file, allow_pickle=True)
        G_diffs_file, chosen_iters_file = params_quality["G_diffs"], params_quality["chosen_iters"]
        M_h_boltz_file, M_J_boltz_file = params_quality["M_h_boltz"], params_quality["M_J_boltz"]
        SD_h_boltz_file, SD_J_boltz_file = params_quality["SD_h_boltz"], params_quality["SD_J_boltz"]
        G_diffs_file[timesteps_idx, pop], chosen_iters_file[timesteps_idx, pop] = G_diffs, chosen_iters
        M_h_boltz_file[timesteps_idx, pop], M_J_boltz_file[timesteps_idx, pop] = M_h_boltz, M_J_boltz
        SD_h_boltz_file[timesteps_idx, pop], SD_J_boltz_file[timesteps_idx, pop] = SD_h_boltz, SD_J_boltz
        np.savez(file, info=params_quality["info"], G_diffs=G_diffs_file, chosen_iters=chosen_iters_file, \
                 M_h_boltz=M_h_boltz_file, M_J_boltz=M_J_boltz_file, SD_h_boltz=SD_h_boltz_file, SD_J_boltz=SD_J_boltz_file, \
                 allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    params_quality = {}
    params_quality["info"] = f"Looking at convergence of G_diff as a function of Boltzmann learning iteration for synthetic data. \
                             Boltzmann learning parameters: eta={eta}, nr_samples={nr_samples}, nr_iterations={nr_iterations}. \
                             neurons = {N}, timesteps = {Timesteps}, seed = timesteps + pop, nr_evals = {nr_evals}. \
                             {pops_per_timesteps} populations/datasets per for each number of samples." 
    params_quality["G_diffs"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["G_diffs"][timesteps_idx, pop] = G_diffs
    params_quality["chosen_iters"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["chosen_iters"][timesteps_idx, pop] = chosen_iters
    params_quality["M_h_boltz"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["M_h_boltz"][timesteps_idx, pop] = M_h_boltz
    params_quality["M_J_boltz"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["M_J_boltz"][timesteps_idx, pop] = M_J_boltz
    params_quality["SD_h_boltz"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["SD_h_boltz"][timesteps_idx, pop] = SD_h_boltz
    params_quality["SD_J_boltz"] = np.full((len(Timesteps), pops_per_timesteps, nr_evals), np.nan) 
    params_quality["SD_J_boltz"][timesteps_idx, pop] = SD_J_boltz
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **params_quality, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
