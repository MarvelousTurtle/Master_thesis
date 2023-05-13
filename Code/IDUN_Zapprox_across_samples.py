import numpy as np
from Functions_Ising_model import *
import fcntl # Only avaliable on Unix platforms
import sys

run = int(sys.argv[1]) - 1
populations = 500
rng = np.random.default_rng(42 + run)

N = 15
samples_start, samples_end = 100, 20000
nr_props = 100
SD_params = 1/np.sqrt(N-1) 

h = rng.normal(0, SD_params, N) 
J = np.zeros((N, N))
J[np.triu_indices(N, 1)] = rng.normal(0, SD_params, int(N*(N-1)/2))
J = (J + J.T) 
J = np.tril(J, k=-1) 

Z = 0
for s in itertools.product([-1, 1], repeat=len(h)):
    s = np.array(s)
    Z += np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))

data = Metropolis_samples(samples_end + 5000, h, J, 0.5, rng)[5000:]

eta = 0.001
nr_samples = 5000
nr_iterations = 30000
h_initial , J_initial = np.zeros(N), np.zeros((N, N)) 

data_props = np.linspace(samples_start, samples_end + 1, nr_props)
Zs_mean = np.zeros(len(data_props))
Zs_median = np.zeros(len(data_props))
Zs_MostSampled = np.zeros(len(data_props))
Zs_hat = np.zeros(len(data_props))
Zs = np.zeros(len(data_props))
for idx, p in enumerate(data_props):
    # Can fit parameters here
    h, J, _, _, = Boltzmann_learning(data[0:int(p)], h_initial, J_initial, eta, nr_iterations, nr_samples, rng)
    # h, J = pseudolikelihood_sklearn(data[0:int(p)]) 
    J = np.tril((J + J.T) / 2, k=-1) # !!
    
    for s in itertools.product([-1, 1], repeat=len(h)):
        s = np.array(s)
        Zs[idx] += np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
    
    sampled_states_data, counts = np.unique(data[0:int(p)], axis=0, return_counts=True)
    dist_data = counts / data[0:int(p)].shape[0]
    e_pair = np.empty(len(sampled_states_data))
    for idx_s, s in enumerate(sampled_states_data):
        e_pair[idx_s] = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
    Zs_mean[idx] = np.mean(e_pair / dist_data)
    Zs_median[idx] = np.median(e_pair / dist_data)
    Zs_MostSampled[idx] = e_pair[np.argmax(dist_data)] / np.max(dist_data)
    # opti = minimize_scalar(lambda Z: np.sum((dist_data - e_pair / Z)**2), method="Brent", bracket=[1, 2]) 
    Zs_hat[idx] = np.sum(e_pair * e_pair) / np.sum(dist_data * e_pair) # opti.x
    
filename = f"Zapprox_samples{samples_start}-{samples_end}_neurons{N}_populatins{populations}.npz" 

try:
    with open(filename, "rb+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        Zapprox = np.load(file, allow_pickle=True)
        Z_f, Zs_f, Zs_hat_f = Zapprox["Z"], Zapprox["Zs"], Zapprox["Zs_hat"]
        Zs_mean_f, Zs_median_f, Zs_MostSampled_f = Zapprox["Zs_mean"], Zapprox["Zs_median"], Zapprox["Zs_MostSampled"]
        Z_f[run], Zs_f[run], Zs_hat_f[run] = Z, Zs, Zs_hat
        Zs_mean_f[run], Zs_median_f[run], Zs_MostSampled_f[run] = Zs_mean, Zs_median, Zs_MostSampled
        np.savez(file, info=Zapprox["info"], Z=Z_f, Zs=Zs_f, Zs_hat=Zs_hat_f, Zs_mean=Zs_mean_f, \
                 Zs_median=Zs_median_f, Zs_MostSampled=Zs_MostSampled_f, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
except FileNotFoundError: 
    Zapprox = {}
    Zapprox["info"] = f"Approximations of Z for {nr_props} points between {samples_start} and {samples_end} samples. \
                        For each point, Boltzmann learning was used to approximate the model parameters and Zs from data. \
                        Boltzmann learning parameters: eta={eta}, nr_samples={nr_samples}, nr_iterations={nr_iterations}. \
                        N={N}, seed=1, nr_populations={populations}."
    Zapprox["Z"] = np.full(populations, np.nan)
    Zapprox["Z"][run] = Z
    Zapprox["Zs"] = np.full((populations, nr_props), np.nan)
    Zapprox["Zs"][run] = Zs
    Zapprox["Zs_hat"] = np.full((populations, nr_props), np.nan)
    Zapprox["Zs_hat"][run] = Zs_hat
    Zapprox["Zs_mean"] = np.full((populations, nr_props), np.nan)
    Zapprox["Zs_mean"][run] = Zs_mean
    Zapprox["Zs_median"] = np.full((populations, nr_props), np.nan)
    Zapprox["Zs_median"][run] = Zs_median
    Zapprox["Zs_MostSampled"] = np.full((populations, nr_props), np.nan)
    Zapprox["Zs_MostSampled"][run] = Zs_MostSampled
    with open(filename, "wb") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        np.savez(file, **Zapprox, allow_pickle=True)
        fcntl.flock(file, fcntl.LOCK_UN)
