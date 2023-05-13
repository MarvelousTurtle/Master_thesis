import numpy as np
import itertools 
from sklearn.linear_model import LogisticRegression

# Sampling Ising model 

def random_state(n, prop_negative, rng=np.random.default_rng()):
    rand = rng.random(n)
    state = np.zeros(n)
    state[rand >= prop_negative] = 1
    state[rand < prop_negative] = -1
    return state

def H(state, h, J): # Hamiltonian / energy 
    J = np.tril(J, k=-1) 
    H = - J @ state @ state.T - h @ state.T
    return H

def Metropolis_samples(timesteps, h, J, s0_prob, rng=np.random.default_rng()):
    n = len(h)
    state = random_state(n, s0_prob, rng) # Can change average proportion of -1 spins
    
    states = np.zeros((timesteps + 1, n), dtype=np.int8)
    H_old = H(state, h, J)
    states[0] = state
    
    for t in range(timesteps):
        state_new = np.copy(state)
        i = rng.integers(0, n)
        state_new[i] *= -1
        
        H_new = H(state_new, h, J)
        deltaH = H_new - H_old 
        
        p = np.minimum(1, np.exp(-deltaH)) # Metropolis-Hastings algorithm 
        if rng.random(1) < p: 
            state[i] *= -1 
            H_old = np.copy(H_new)
        states[t+1] = state
                
    return states

# Approximating h and J

def means_corrs_data(data):
    t, n = np.shape(data)
    data = np.array(data, dtype=np.float64) 
    means_data = np.mean(data, axis=0)
    corrs_data = np.tensordot(data, data, axes=(0, 0)) / data.shape[0] 
    return means_data, corrs_data

def Boltzmann_learning(data, h_initial, J_initial, eta, N_iterations, MC_samples, rng=np.random.default_rng()): 
    t, n = np.shape(data)
    means_data, corrs_data = means_corrs_data(data) 
    
    h = h_initial 
    J = J_initial 
    hs, Js = np.empty((N_iterations + 1, n)), np.empty((N_iterations + 1, n, n)) # only for plotting
    hs[0], Js[0] = h, J # only for plotting
    for i in range(N_iterations):
        samples = Metropolis_samples(MC_samples + 1000, h, J, 0.5, rng)[1000:]
        means_samples, corrs_samples = means_corrs_data(samples) 
        h += eta * (means_data - means_samples)
        J += eta * (corrs_data - corrs_samples) 
        hs[i + 1], Js[i + 1] = h, J 

    return h, J, hs[1:], Js[1:]

def pseudolikelihood_sklearn(data): # This is faster and better than pseudolikelihood() in most cases
    t, n = np.shape(data)
    h_PL, J_PL = np.zeros(n), np.zeros((n, n))
    for i in range(n):
        # This does not work if a neuron never fires
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=10000, tol=1e-6).fit(np.delete(data, i, axis=1), data[:, i])
        h_PL[i] = model.intercept_[0]
        J_PL[i] = np.insert(model.coef_[0], i, 0)
    J_PL = (J_PL + J_PL.T) / 2
    h_PL, J_PL = h_PL / 2, J_PL / 2 
    return h_PL, J_PL

def pseudolikelihood(data, h_initial, J_initial, eta, N_iterations): 
    t, n = np.shape(data)
    means_data, corrs_data = means_corrs_data(data) 
    
    h = h_initial 
    J = J_initial 
    hs, Js = np.empty((N_iterations + 1, n)), np.empty((N_iterations + 1, n, n)) # only for plotting
    hs[0], Js[0] = h, J # only for plotting
    for i in range(N_iterations):
        means_model = np.sum(np.tanh(h[None, :] + data @ J), axis=0) / data.shape[0]
        corrs_model = data.T @ np.tanh(h[None, :] + data @ J) / data.shape[0]
        np.fill_diagonal(J, 0) 
        h += eta * (means_data - means_model)
        J += eta * (corrs_data - corrs_model) 
        hs[i + 1], Js[i + 1] = h, J 

    return h, J, hs[1:], Js[1:]

def nMF(data): 
    t, n = np.shape(data)
    m, C = means_corrs_data(data)
    C -= np.outer(m, m)
        
    J = - np.linalg.inv(C)
    np.fill_diagonal(J, 0) 
    h = np.arctanh(m) - np.sum(J * m[None, :], axis=1) 

    return h, J
    
def TAP(data):
    t, n = np.shape(data)
    m, C = means_corrs_data(data)
    C -= np.outer(m, m)
    C_inv = np.linalg.inv(C) 
    
    J = np.zeros((n, n), dtype=np.cdouble) # Allows for complex roots
    r, c = np.triu_indices(n, 1)
    for i, j in zip(r, c):
        roots = np.roots((2 * m[i] * m[j], 1, C_inv[i, j]))
        J[i, j] = roots[np.argmin(np.abs(roots + C_inv[i, j]))] # just taking the second root is faster
        
    J = J + J.T
    h = np.arctanh(m) - np.sum(J * m[None, :], axis=1) + m * np.sum(J**2 * (1 - m[None, :]**2), axis=1)
    
    return h, J

def IP(data):
    t, n = np.shape(data)
    m, C = means_corrs_data(data)
    C -= np.outer(m, m)
    
    np.fill_diagonal(C, 0)
    J = np.log(((np.outer(1 + m, 1 + m) + C) * (np.outer(1 - m, 1 - m) + C)) / \
               ((np.outer(1 - m, 1 + m) - C) * (np.outer(1 + m, 1 - m) - C))) / 4
    np.fill_diagonal(J, 0)
    h = np.log((1 + m) / (1 - m)) / 2 + \
        np.sum(np.log(((1 - m[None, :] - C + m[:, None] - m[None, :] * m[:, None]) / (1 + m[:, None])) / \
                      ((1 - m[None, :] + C - m[:, None] + m[None, :] * m[:, None]) / (1 - m[:, None]))), axis=1) / 2 + \
        np.sum(J, axis=1)
    
    return h, J

def SM(data): 
    t, n = np.shape(data)
    m, C = means_corrs_data(data)
    C -= np.outer(m, m)
    C_inv = np.linalg.inv(C) 
    np.fill_diagonal(C, 0)
    J_IP = np.log(((np.outer(1 + m, 1 + m) + C) * (np.outer(1 - m, 1 - m) + C)) / \
                  ((np.outer(1 - m, 1 + m) - C) * (np.outer(1 + m, 1 - m) - C))) / 4
        
    J = - C_inv + J_IP - C / (np.outer(1 - m**2, 1 - m**2) - C**2)
    np.fill_diagonal(J, 0)
    
    # h from equation 21 in Sessak & Monasson (2009)
    L = 1 - m**2
    K = C / np.outer(L, L)
    exl_diag = ~np.eye(n, n, 0, dtype=bool)
    smaller_idx, larger_idx = np.triu_indices(n, 1)
    h = np.log((1 + m) / (1 - m)) / 2 \
        - J @ m \
        + np.sum(K**2 * np.outer(m, L) * exl_diag, axis=1) \
        - 2 * (1 + 3 * m**2) / 3 * np.sum(K**3 * (m * L)[None, :] * exl_diag, axis=1) \
        - 2 * m * np.sum(K[:, smaller_idx] * K[smaller_idx, larger_idx][None, :] * K[larger_idx, :].T * (L[smaller_idx] * L[larger_idx])[None, :], axis=1) \
        + 2 * m * np.sum(K[smaller_idx, :][:, :, None] * K[:, larger_idx].T[:, :, None] * K[larger_idx, :][:, None, :] * K[:, smaller_idx].T[:, None, :] * \
                          L[smaller_idx][:, None, None] * L[larger_idx][:, None, None] * L[:][None, :, None], axis=(0, 1)) \
        + m * np.sum(K**4 * L[None, :] * (1 + (m**2)[:, None] + 3 * (m**2)[None, :] + 3 * (m**2)[:, None] * (m**2)[None, :]), axis=1) \
        + np.sum((m[:, None, None] * (K**2)[None, :, :] * (K**2).T[:, None, :] * L[None, :, None] * (L**2)[None, None, :]) * exl_diag[:, :, None], axis=(1, 2))
           
    return h, J

# Quality measure functions

def entropy_plugin(samples):
    _, counts = np.unique(samples, axis=0, return_counts=True)
    dist = counts / samples.shape[0]
    S = - np.sum(dist * np.log2(dist)) 
    
    return S

def entropy_from_params(h, J):
    # This takes very slightly longer than the above function, but reduces
    # the space complexity drastically. 
    J = np.tril(J, k=-1) 
    sum1, Z = 0, 0
    for s in itertools.product([-1, 1], repeat=len(h)):
        s = np.array(s)
        temp = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        sum1, Z = sum1 + temp * np.log2(temp), Z + temp
    S = - sum1 / Z + np.log2(Z)
    
    return S

def KLdivergence_plugin(samples, h, J):
    sampled_states, counts = np.unique(samples, axis=0, return_counts=True)
    dist = counts / samples.shape[0]
    J = np.tril(J, k=-1)  
    sum1, Z = 0, 0
    for k, s in enumerate(itertools.product([-1, 1], repeat=len(h))):
        s = np.array(s)
        temp = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        sampled_state = np.where((s == sampled_states[:k+1]).all(axis=1))[0]
        if sampled_state.size == 1:
            prob = np.squeeze(dist[sampled_state])
            sum1 += prob * np.log2(temp)
        Z += temp
    D = np.sum(dist * np.log2(dist)) - sum1 + np.log2(Z)
    return D

def KLdivergence_from_params(h1, J1, h2, J2):
    J1, J2 = np.tril(J1, k=-1), np.tril(J2, k=-1)  
    sum1, sum2, Z1, Z2 = 0, 0, 0, 0
    for s in itertools.product([-1, 1], repeat=len(h1)):
        s = np.array(s)
        temp1 = np.exp(np.sum(J1 * s[None, :] * s[:, None]) + np.sum(h1 * s))
        temp2 = np.exp(np.sum(J2 * s[None, :] * s[:, None]) + np.sum(h2 * s))
        sum1, sum2 = sum1 + temp1 * np.log2(temp1), sum2 + temp1 * np.log2(temp2)
        Z1, Z2 = Z1 + temp1, Z2 + temp2
    D = sum1 / Z1 - sum2 / Z1 + np.log2(Z2) - np.log2(Z1)
    return D

def G_plugin(samples, h=None, J=None):
    if h is None and J is None:
        h, J = pseudolikelihood_sklearn(samples) # This is here so that the finite_samples_correction() function works generally.
    
    sampled_states, counts = np.unique(samples, axis=0, return_counts=True)
    dist = counts / samples.shape[0]
    J = np.tril(J, k=-1)  
    h_ind = np.arctanh(np.mean(samples, axis=0))
    sum1, sum2, sum3, Z_pair, Z_ind = 0, 0, 0, 0, 0
    for k, s in enumerate(itertools.product([-1, 1], repeat=len(h))):
        s = np.array(s)
        temp_pair = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        temp_ind = np.exp(np.sum(h_ind * s))
        sampled_state = np.where((s == sampled_states[:k+1]).all(axis=1))[0]
        if sampled_state.size == 1:
            p_data = np.squeeze(dist[sampled_state])
            sum1, sum2, sum3 = sum1 + p_data * np.log2(p_data), sum2 + p_data * np.log2(temp_pair), sum3 + p_data * np.log2(temp_ind)
        Z_pair, Z_ind = Z_pair + temp_pair, Z_ind + temp_ind
    KL_pair, KL_ind = sum1 - sum2 + np.log2(Z_pair), sum1 - sum3 + np.log2(Z_ind)
    G = 1 - KL_pair / KL_ind
    return G, KL_pair, KL_ind

def G_plugin_RC(samples, h=None, J=None):
    if h is None and J is None:
        h, J = pseudolikelihood_sklearn(samples) # This is here so that the finite_samples_correction() function works generally.
    
    sampled_states, counts = np.unique(samples, axis=0, return_counts=True)
    dist = counts / samples.shape[0]
    
    J = np.tril(J, k=-1)  
    sum1, sum2, sum3, Z_pair, Z_ind = 0, 0, 0, 0, 0
    for k, s in enumerate(itertools.product([-1, 1], repeat=len(h))):
        s = np.array(s)
        temp_pair = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        temp_ind = np.exp(np.sum(h * s))
        sampled_state = np.where((s == sampled_states[:k+1]).all(axis=1))[0]
        if sampled_state.size == 1:
            p_data = np.squeeze(dist[sampled_state])
            sum1, sum2, sum3 = sum1 + p_data * np.log2(p_data), sum2 + p_data * np.log2(temp_pair), sum3 + p_data * np.log2(temp_ind)
        Z_pair, Z_ind = Z_pair + temp_pair, Z_ind + temp_ind
    KL_pair, KL_ind = sum1 - sum2 + np.log2(Z_pair), sum1 - sum3 + np.log2(Z_ind)
    G = 1 - KL_pair / KL_ind
    return G, KL_pair, KL_ind

def G_plugin_entropies(samples, h, J):
    Zs_ind = np.exp(h) + np.exp(-h)
    S_ind = - np.sum(np.exp(h) / Zs_ind * np.log2(np.exp(h) / Zs_ind) + np.exp(-h) / Zs_ind * np.log2(np.exp(-h) / Zs_ind))
    S_data = entropy_plugin(samples) 
    S_pair = entropy_from_params(h, J)
    G, KL_pair, KL_ind = (S_ind - S_pair) / (S_ind - S_data), S_pair - S_data, S_ind - S_data

    return G, KL_pair, KL_ind

def finite_samples_correction(samples, func, params_func=[], rng=np.random.default_rng()):
    rng.shuffle(samples, axis=0) 
    nr_samples = (np.array((1, 1/2, 1/5, 1/10)) * len(samples)).astype(int) # can change this
    plugin_values = []
    for T in nr_samples:
        plugin_equal_T = []
        for s in range(1, len(samples) // T + 1):
            subsample = samples[(s - 1) * T : s * T]
            plugin_equal_T.append(func(subsample, *params_func))  
        plugin_values.append(np.mean(plugin_equal_T, axis=0))
    coeffs = np.polyfit(1 / nr_samples, np.array(plugin_values), deg=2)
        
    return coeffs[2]

def Ghat(data, h=None, J=None, L_ratio=False):
    if h is None and J is None:
        h, J = pseudolikelihood_sklearn(data) # This is here so that the finite_samples_correction() function works generally.
    
    sampled_states_data, counts_data = np.unique(data, axis=0, return_counts=True)
    dist_data = counts_data / data.shape[0]

    J = np.tril(J, k=-1) 
    e_pair = np.empty(len(sampled_states_data))
    dist_ind = np.empty(len(sampled_states_data))
    h_ind = np.arctanh(np.mean(data, axis=0))
    sum_pair = 0
    for idx, s in enumerate(sampled_states_data):
        e_pair[idx] = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        sum_pair += dist_data[idx] * np.log2(e_pair[idx])
        dist_ind[idx] = np.prod(np.exp(s * h_ind) / (np.exp(h_ind) + np.exp(-h_ind)))
    KL_ind = np.sum(dist_data * np.log2(dist_data / dist_ind))
    
    Z_approx_pair = np.max((np.sum(e_pair * e_pair) / np.sum(dist_data * e_pair), np.sum(e_pair)))
    
    KL_pair = np.sum(dist_data * np.log2(dist_data)) - sum_pair + np.log2(Z_approx_pair)
    G = 1 - KL_pair / KL_ind
    
    if L_ratio:
        L_pair = np.sum((dist_data - e_pair / Z_approx_pair)**2)
        L_ind = np.sum((dist_data - dist_ind)**2)
        L_ratio = 1 - L_pair / L_ind
        return G, KL_pair, KL_ind, L_ratio

    return G, KL_pair, KL_ind

def Ghat_RC(data, h=None, J=None, L_ratio=False):
    if h is None and J is None:
        h, J = pseudolikelihood_sklearn(data) # This is here so that the finite_samples_correction() function works generally.
    
    sampled_states_data, counts_data = np.unique(data, axis=0, return_counts=True)
    dist_data = counts_data / data.shape[0]

    J = np.tril(J, k=-1) 
    e_pair = np.empty(len(sampled_states_data))
    dist_ind = np.empty(len(sampled_states_data))
    sum_pair = 0
    for idx, s in enumerate(sampled_states_data):
        e_pair[idx] = np.exp(np.sum(J * s[None, :] * s[:, None]) + np.sum(h * s))
        sum_pair += dist_data[idx] * np.log2(e_pair[idx])
        dist_ind[idx] = np.prod(np.exp(s * h) / (np.exp(h) + np.exp(-h)))
    KL_ind = np.sum(dist_data * np.log2(dist_data / dist_ind))

    Z_approx_pair = np.max((np.sum(e_pair * e_pair) / np.sum(dist_data * e_pair), np.sum(e_pair)))
    
    KL_pair = np.sum(dist_data * np.log2(dist_data)) - sum_pair + np.log2(Z_approx_pair)
    G = 1 - KL_pair / KL_ind
    
    if L_ratio:
        L_pair = np.sum((dist_data - e_pair / Z_approx_pair)**2)
        L_ind = np.sum((dist_data - dist_ind)**2)
        L_ratio = 1 - L_pair / L_ind
        return G, KL_pair, KL_ind, L_ratio

def corrs3(data, connected=False): 
    t, n = data.shape
    means = np.mean(data, axis=0)
    corrs3 = np.zeros(int(n * (n - 1) * (n - 2) / 6))
    for idx, triplet in enumerate(itertools.combinations(np.arange(0, n), 3)):
        triplet = np.array(triplet)
        if connected:
            corrs3[idx] = np.mean((data[:, triplet[0]] - means[triplet[0]]) * \
                                  (data[:, triplet[1]] - means[triplet[1]]) * \
                                  (data[:, triplet[2]] - means[triplet[2]]))
        else: 
            corrs3[idx] = np.mean(data[:, triplet[0]] * data[:, triplet[1]] * data[:, triplet[2]])
    return corrs3

def SimulSpikes(data): 
    t, n = data.shape
    n_spikes, counts_spikes = np.unique(np.count_nonzero(data + 1, axis=1), return_counts=True)
    counts = np.zeros(n + 1)
    counts[n_spikes] = counts_spikes
    return counts

# Spike times --> Spike train

def time_binning(Stimes, interval, timebin):
    n_bins = int(np.round((interval[1] - interval[0]) / timebin, 0)) 
    Strain = np.empty((n_bins, len(Stimes)))
    for n in range(len(Stimes)):
        Strain[:, n], _ = np.histogram(Stimes[n], n_bins, range=interval)
    Strain[Strain==0] = -1
    Strain[Strain>0] = 1
    return Strain

def subpopulation_spike_trains(spike_trains, N, binsize, rng=np.random.default_rng()):
    
    all_indices = np.arange(0, spike_trains.shape[1])
    means = np.count_nonzero(spike_trains + 1, axis=0) / (spike_trains.shape[0] * binsize) # firing rates
    
    popu = np.zeros(N, dtype=int)
    weights_means = means / np.sum(means)
    chosen_idx = rng.choice(np.arange(0, means.shape[0]), 1, p = weights_means)
    chosen_mean = means[chosen_idx]
    same_mean_idx = np.argwhere(means == chosen_mean) 
    popu[0:len(same_mean_idx)] = same_mean_idx[:, 0]
    all_indices = np.delete(all_indices, same_mean_idx)
    means = np.delete(means, same_mean_idx)
    weights_popu = (1 / np.abs(chosen_mean - means))**3 # might want to use **2 or **4 instead to decrease all SDs 
    weights_popu = weights_popu / np.sum(weights_popu)
    popu[len(same_mean_idx):N] = rng.choice(all_indices, N - len(same_mean_idx), p = weights_popu, replace=False)

    return popu

def subpopulation_spike_times(spike_times, N, interval, rng=np.random.default_rng()):
    
    all_indices = np.arange(0, len(spike_times))
    means = np.array([len(spike_times[n]) for n in range(len(spike_times))]) / (interval[1] - interval[0])
    
    popu = np.zeros(N, dtype=int)
    weights_means = means / np.sum(means)
    chosen_idx = rng.choice(np.arange(0, means.shape[0]), 1, p = weights_means)
    chosen_mean = means[chosen_idx]
    same_mean_idx = np.argwhere(means == chosen_mean) 
    popu[0:len(same_mean_idx)] = same_mean_idx[:, 0]
    all_indices = np.delete(all_indices, same_mean_idx)
    means = np.delete(means, same_mean_idx)
    weights_popu = (1 / np.abs(chosen_mean - means))**3 # might want to use **2 or **4 instead to decrease all SDs 
    weights_popu = weights_popu / np.sum(weights_popu)
    popu[len(same_mean_idx):N] = rng.choice(all_indices, N - len(same_mean_idx), p = weights_popu, replace=False)

    return popu
