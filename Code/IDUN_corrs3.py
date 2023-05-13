import numpy as np
from Functions_Ising_model import *

N = 100
binsize = 0.02
rng = np.random.default_rng(N)

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

subpop = rng.choice(len(spike_times), N, replace=False)
samples_data = time_binning(spike_times[subpop], interval, binsize)

h, J = pseudolikelihood_sklearn(samples_data)

nr_samples = samples_data.shape[0]
samples_pair = Metropolis_samples(nr_samples, h, J, 0.5, rng)

corrs3_data = corrs3(samples_data)
corrs3_pair = corrs3(samples_pair)

filename = f"corrs3_{N}neurons.npz" 
info = f"Third-order correlations from a random subpopulation of {N} neurons (from Neuropixel data) \
         and equally many samples from the pairwise model inferred with pseudolikelihood. \
         Pseudolikelihood was used to approximate the parameters. Seed = {N}, binsize = {binsize}."         
np.savez(filename, info=info, corrs3_data=corrs3_data, corrs3_pair=corrs3_pair, allow_pickle=True)
