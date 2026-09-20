import numpy as np
import pandas as pd
import yfinance as yf
import itertools
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta

def load_data(currencies: list[str], start: int, end: int) -> list[pd.DataFrame]: #currency = "BTC-USD" for example

    try:
        today = datetime.now()
        train_start = (today - timedelta(days = start)).strftime("%Y-%m-%d")
        train_end = (today - timedelta(days = end)).strftime("%Y-%m-%d")

        #We get a raw data, because it is possible to be void lines for some dates

        raw_datas = [yf.Ticker(currency).history(start = train_start, end = train_end).ffill().bfill() for currency in currencies] #stored as a list of pandas DataFrame objects

        #From here, we can use any pandas DataFrame module

        common_index = raw_datas[0].index

        for df in raw_datas[1:]:
            common_index = common_index.intersection(df.index)

        if len(common_index) == 0:
            return "There are no common valid dates between given currencies."

        datas = [df.loc[common_index] for df in raw_datas]
        return datas
    
    except Exception as e:
        print(f"Invalid input given: {e}.")

def calculate_states(data: list[pd.DataFrame], n_states: int) -> pd.DataFrame:

    if len(data) == 0 or not isinstance(data[0], pd.DataFrame):
        raise ValueError("Invalid input given.")

    df = pd.DataFrame(index=data[0].index) # Suppose all dataframes have the same index, we can use the index of the first dataframe

    for i in range(len(data)):
        if data[i].empty:
            print(f"DataFrame at index {i} is empty. Please check the input data.")
            return
        
        df[f'Change {i}'] = (data[i]['Close'] - data[i]['Open']) / data[i]['Open'] * 100

        conditions = [
            df[f'Change {i}'] < -1, #Bear
            (df[f'Change {i}'] >= -1) & (df[f'Change {i}'] <= 1), #Stagnate
            df[f'Change {i}'] > 1 #Bull
        ]
        df[f'State {i}'] = np.select(conditions, list(range(n_states))) # 0 = Bear, 1 = Stagnate, 2 = Bull

    change_cols = [f'Change {i}' for i in range(len(data))]
    df['Change'] = df[change_cols].values.tolist()

    state_cols = [f'State {i}' for i in range(len(data))]
    df['State'] = df[state_cols].values.tolist()

    print(df.head()[['Change', 'State']])

    return df

#Initializing the transition matrix A

def initialize_transition_matrix(df: pd.DataFrame, n: int, dim: int) -> np.ndarray:

    n_states = n**dim
    elements = list(range(n))
    vectors = list(itertools.product(elements, repeat = dim))
    vectors = [list(x) for x in vectors]

    A = np.zeros((n_states, n_states)) + 1e-6 # to avoid division by zero, we add a small value to the matrix


    for index1, i in enumerate(vectors):
        for index2, j in enumerate(vectors):
            for k in range(len(df['State'])):
                if k > 0 and df['State'].iloc[k] == j and df['State'].iloc[k-1] == i:
                    A[index1, index2] += 1


    row_sums = A.sum(axis = 1, keepdims = True)

    A = np.divide(A, row_sums, where = row_sums != 0, out = np.zeros_like(A)) 

    return A

#Initializing the emission matrix B

def initialize_means_covs(df: pd.DataFrame, n1: int, dim1: int) -> tuple[np.ndarray, np.ndarray]:


    n1_states = n1**dim1

    elements = list(range(n1))
    vectors = list(itertools.product(elements, repeat = dim1))
    vectors = [list(x) for x in vectors]

    means = np.zeros((n1_states, dim1))
    covs = np.zeros((n1_states, dim1, dim1))

    for index, vector in enumerate(vectors):
        mask = df['State'].apply(lambda x: x == vector)
        state_datas_matrix = np.array(df[mask]['Change'].tolist())
        if len(state_datas_matrix) > 1:
            means[index] = np.mean(state_datas_matrix, axis = 0)
            covs[index] = np.cov(state_datas_matrix, rowvar = False)
            print(f"State: {vector}, Mean: {means[index]}, Covariance: {covs[index]}")
        else:
            means[index] = np.zeros(dim1)
            covs[index] = np.eye(dim1)

    return means, covs

def initialize_distribution(A: np.ndarray) -> np.ndarray: #stationary distribution, by a theorem, we can find the stationary distribution by solving the equation pi * A = pi, where pi is the stationary distribution vector

    K = A.shape[0]
    M = A.T - np.eye(K)
    M[-1, :] = 1

    b = np.zeros(K)
    b[-1] = 1

    pi = np.linalg.solve(M, b)
    return pi


def initialize_emission_matrix(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:

    X = np.asarray(X)

    T, D = X.shape #T and D as time and dimension
    n_states = means.shape[0] # number of hidden states

    B = np.zeros((T, n_states))

    for j in range(n_states):
        B[:, j] = multivariate_normal.pdf(X, mean = means[j], cov = (covs[j] + 1e-6 * np.eye(D)))

    #Previous cycle does the following in a more optimal way:
    #for i in range(T):
    #    for j in range(n_states):
    #        B[i, j] = multivariate_normal.pdf(X[i], mean = means[j], cov = covs[j])

    return B

def generate_random_params(X: np.ndarray, n_states: int): #not needed for the algorithm, just for showing how much better is to give a good initialization
    X = np.asarray(X)
    T, D = X.shape
    
    rand_idx = np.random.choice(T, size=n_states, replace=False)
    means_rand = X[rand_idx].copy()
    
    global_cov = np.cov(X, rowvar=False)
    covs_rand = np.array([
        global_cov * np.random.uniform(0.5, 5) + 1e-5 * np.eye(D)
        for _ in range(n_states)
    ])
    
    A_rand = np.random.dirichlet(np.ones(n_states), size=n_states)
    pi_rand = np.random.dirichlet(np.ones(n_states))
    
    return A_rand, pi_rand, means_rand, covs_rand

#Instance running of code:


datas = load_data(["BTC-USD", "ETH-USD"], 90, 30) # Load data for Bitcoin in USD
states = calculate_states(datas, 3) # Calculate states for Bitcoin and Ethereum with 3 states
A = initialize_transition_matrix(states, 3, 2) # Initialize transition matrix A
pi = initialize_distribution(A)
means_covs = initialize_means_covs(states, 3, 2) # Initialize means and covariances for emission matrix B
print(A)
print(pi) # Initialize stationary distribution