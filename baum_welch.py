import numpy as np
import pandas as pd
import yfinance as yf
import itertools
from scipy.stats import multivariate_normal
from initialize import *

#The following code implements Baum-Welch when observations has continuous distribution.
#Observations only occur in recalibration of emission probabilities.

input = ["BTC-USD", "ETH-USD"]

datas = load_data(input, 90, 30)
states = calculate_states(datas, 3) # Calculate states for Bitcoin and Ethereum with 3 states
A = initialize_transition_matrix(states, 3, 2) # Initialize transition matrix A
pi = initialize_distribution(A) # Initialize stationary distribution
means_covs = initialize_means_covs(states, 3, 2) # Initialize means and covariances for emission matrix B


#Expectation step of Baum-Welch algorithm: getting alpha, beta, gamma, ksi.

def forward_scaled(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    T, N = B.shape

    alpha = np.zeros((T, N))
    c_alpha = np.zeros(T) # scaling factors

    alpha_hat_0 = pi * B[0]
    c_alpha[0] = 1 / (np.sum(alpha_hat_0) + 1e-300)
    alpha[0] = alpha_hat_0 * c_alpha[0]

    for t in range(1, T):
        alpha_hat_t = (alpha[t-1] @ A) * B[t]
        c_alpha[t] = 1 / (np.sum(alpha_hat_t) + 1e-300)

        alpha[t] = alpha_hat_t * c_alpha[t]

    return alpha, c_alpha

def backward_scaled(A: np.ndarray, B: np.ndarray, c_alpha: np.ndarray) -> np.ndarray:

    T, N = B.shape

    beta = np.zeros((T, N))
    beta[T-1] = c_alpha[T-1]

    for t in range(T-2, -1, -1):
        beta_hat_t = A @ (B[t+1] * beta[t+1])

        beta[t] = beta_hat_t * c_alpha[t]

    return beta

def gamma_belief(alpha: np.ndarray, c_alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:

    T, N = alpha.shape
    gamma = np.zeros((T, N))

    for t in range(T):
        gamma[t] = alpha[t] * beta[t] / c_alpha[t] # bel[t]
    return gamma

def ksi_belief(alpha: np.ndarray, c_alpha: np.ndarray, beta: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:

    T, N = alpha.shape

    ksi = np.zeros((T-1, N, N))

    for t in range(0, T-1):
        for i in range(N):
            for j in range(N):
                ksi[t, i, j] = alpha[t, i] * A[i, j] * B[t+1, j] * beta[t+1, j] * c_alpha[t+1]

    return ksi

#Maximization step of Baum-Welch: gettin pi_new, A_new, B_new

def recalibrate_transition_matrix(ksi: np.ndarray, gamma: np.ndarray):
    T, N = gamma.shape

    A_new = np.zeros((N, N))

    gamma_sum = np.sum(gamma[:-1], axis = 0)

    for i in range(N):

        if gamma_sum[i] < 1e-12:
            A_new[i, :] = 1.0 / N
        else:

            for j in range(N):
                #(expected number of transitions from S_i to S_j) / (expected number of transitions from S_i)
                A_new[i, j] = np.sum(ksi[:, i, j], axis = 0) / gamma_sum[i]

    A_new = np.nan_to_num(A_new, nan = 1.0/N)
    A_new = A_new / np.sum(A_new, axis=1, keepdims=True)

    return A_new

#Before recalibrating B, we need to recalibrate means and covs.

def recalibrate_means_covs(X: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    X = np.asarray(X)

    T, D = X.shape
    n_states = gamma.shape[1]

    means_new = np.zeros((n_states, D))
    covs_new = np.zeros((n_states, D, D))

    gamma_sum_total = np.sum(gamma, axis = 0) # Averages times spent there for every state

    min_cov = 1e-6

    for j in range(n_states):

        if gamma_sum_total[j] < 1e-6:
            covs_new[j] = min_cov * np.eye(D)
            continue

        means_new[j] = np.sum(gamma[:, j, np.newaxis] * X, axis = 0) / gamma_sum_total[j]

        diff_j = X - means_new[j]
        cov_j = np.zeros((D, D))

        cov_j = np.einsum('t, ti, tj -> ij',gamma[:, j], diff_j, diff_j) / gamma_sum_total[j]
        #alternatively covs_new[j] = ((diff_j * gamma[:, j, np.newaxis]).T @ diff_j) / gamma_sum_total[j] + 1e-6 * np.eye(D)

        covs_new[j] = cov_j + min_cov * np.eye(D)

    means_new = np.nan_to_num(means_new, nan=0.0, posinf=0.0, neginf=0.0)
    covs_new = np.nan_to_num(covs_new, nan=0.0, posinf=0.0, neginf=0.0)

    return means_new, covs_new

def recalibrate_emission_matrix(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:

    """
    X: (T, D)
    means: (n_states, D)
    covs: (n_states, D, D)
    """

    T, D = X.shape
    n_states = means.shape[0]

    B_new = np.zeros((T, n_states))

    for j in range(n_states):
        B_new[:, j] = multivariate_normal.pdf(X, mean = means[j], cov = covs[j], allow_singular = True)

    B_new = np.nan_to_num(B_new, nan = 1e-300, posinf=1.0, neginf=1e-300)
    B_new[B_new == 0] = 1e-300

    return B_new


#After all, we can put together the algorithm logic

def baum_welch_algorithm(X: np.ndarray, states: pd.DataFrame, n_states: int, dim: int, has_initialization: bool, n_iter: int = 50):

    X = np.asarray(X)

    if has_initialization:
    #Initializitaion:
        A = initialize_transition_matrix(states, n_states, dim)
        pi = initialize_distribution(A)
        means, covs = initialize_means_covs(states, n_states, dim)

    else:
        A, pi, means, covs = generate_random_params(X, n_states)

    B = initialize_emission_matrix(X, means, covs)

    alpha, c_alpha = forward_scaled(A, B, pi)

    log_likelihood = -np.sum(np.log(c_alpha + 1e-300))

    #Expectation-Maximization cycle:

    for iteration in range(n_iter):

        #Expectation step:

        alpha, c_alpha = forward_scaled(A, B, pi)

        #To see converging log-likelihood values

        log_likelihood = -np.sum(np.log(c_alpha + 1e-300))
        #print(f"At iteration {iteration}, log-likelihood is {log_likelihood}")


        beta = backward_scaled(A, B, c_alpha)
        gamma = gamma_belief(alpha, c_alpha, beta)
        ksi = ksi_belief(alpha, c_alpha, beta, A, B)

        #Maximization step:

        A = recalibrate_transition_matrix(ksi, gamma)
        means, covs = recalibrate_means_covs(X, gamma)
        B = recalibrate_emission_matrix(X, means, covs)

    return A, pi, means, covs, B, log_likelihood



