import numpy as np
from Functions_Ising_model import *

binsize = 0.02 
N = 100
rng = np.random.default_rng(N)

with open("Data/Rat 3D Tracking & E-Phys KISN 2020 Dataset/concatenated_spike_times_26471_bank0_495neurons_6sessions.npz", "rb") as file: 
    data = np.load(file, allow_pickle=True)
    spike_times, interval = data["spike_times"], data["interval"]

neuron_indices = subpopulation_spike_times(spike_times, N, interval, rng)
spike_times = spike_times[neuron_indices] # overwriting to save space
spike_trains = time_binning(spike_times, interval, binsize)

eta = 0.01
nr_samples = 50000
nr_iterations = 80000
h_initial, J_initial = np.zeros(N), np.zeros((N, N)) 
h_boltz, J_boltz, hs_boltz, Js_boltz = Boltzmann_learning(spike_trains, h_initial, J_initial, eta, nr_iterations, nr_samples, rng)
# M_h_boltz, M_J_boltz = np.mean(hs_boltz, axis=1), np.mean(Js_boltz, axis=(1, 2))
# SD_h_boltz, SD_J_boltz = np.std(hs_boltz, axis=1), np.std(Js_boltz, axis=(1, 2))
# nr_evals = 10000
# chosen_iters = np.rint(np.linspace(0, nr_iterations - 1, nr_evals)).astype("int64")
# hs_boltz, Js_boltz = hs_boltz[chosen_iters], Js_boltz[chosen_iters]
# means_diff, corrs_diff = means_diff[chosen_iters], corrs_diff[chosen_iters]

h_PL, J_PL = pseudolikelihood_sklearn(spike_trains)
h_nMF, J_nMF = nMF(spike_trains)
h_TAP, J_TAP = TAP(spike_trains)
h_IP, J_IP = IP(spike_trains)
h_SM, J_SM = SM(spike_trains)

filename = f"boltz_vs_approx_{N}neurons_{binsize}binsize.npz" 
info = f"Parameter approximation using Boltzmann learning versus pseudolikelihood, nMF, TAP, IP, and SM. \
         Boltzmann learning parameters: eta={eta}, nr_samples={nr_samples}, nr_iterations={nr_iterations}. \
         neurons = {N}, binsize = {binsize}, seed = {N}. "
         # Ony stores hs_boltz, Js_boltz, means_diff, and corrs_diff for {nr_evals} iterations to save space \
         # (np.rint(np.linspace(0, nr_iterations - 1, nr_evals)).astype('int64'))."         
# np.savez(filename, info=info, h_boltz=h_boltz, J_boltz=J_boltz, h_PL=h_PL, J_PL=J_PL, neuron_indices=neuron_indices, \
#          M_h_boltz=M_h_boltz, M_J_boltz=M_J_boltz, SD_h_boltz=SD_h_boltz, SD_J_boltz=SD_J_boltz, allow_pickle=True)
np.savez(filename, info=info, h_boltz=h_boltz, J_boltz=J_boltz, h_PL=h_PL, J_PL=J_PL, \
         h_nMF=h_nMF, J_nMF=J_nMF, h_TAP=h_TAP, J_TAP=J_TAP, h_IP=h_IP, J_IP=J_IP, h_SM=h_SM, J_SM=J_SM, \
         neuron_indices=neuron_indices, hs_boltz=hs_boltz, Js_boltz=Js_boltz, allow_pickle=True)
