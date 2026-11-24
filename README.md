# Hidden Markov Model for Crypto Market Regimes

A multivariate Gaussian Hidden Markov Model fitted with the **scaled Baum–Welch (EM) algorithm**, implemented from scratch in NumPy and SciPy.
It models the joint daily behaviour of BTC-USD and ETH-USD as hidden market regimes.

## The model

- Each coin's daily open-to-close change (in %) is labelled **bear** (below −1%), **flat** or **bull** (above +1%), which gives 3 × 3 = **9 joint hidden states**.
- Observations are the two daily changes, modelled with a 2-dimensional Gaussian in each state.
- Parameters: transition matrix, initial distribution, and a mean vector and covariance matrix for each state.

## Smart initialization vs. random starts

EM only finds a local optimum, so the starting point matters. Instead of random parameters, the model is initialized from a labelled **60-day training window**:

- the transition matrix is estimated from the observed regime changes,
- the initial distribution is its stationary distribution,
- the means and covariances are estimated from the days in each regime.

<!-- Add a plot here, for example the log-likelihood over EM iterations for both starts:
![Log-likelihood](loglik.png) -->

The scaling of the forward and backward variables prevents numerical underflow on long sequences.
The covariance updates are regularized to avoid singular matrices, so the log-likelihood is not strictly monotone.

## Files

| File | Description |
|---|---|
| `initialize.py` | Data download (yfinance), regime labelling, and initialization of the transition matrix, stationary distribution, means and covariances |
| `baum_welch.py` | Scaled forward–backward algorithm and the Baum–Welch (EM) updates |
| `show.py` | Compares the initialized model with random starts |
| `HMM_application.pdf` | Theory, derivations and results, in Hungarian |

## How to run

```bash
pip install -r requirements.txt
python show.py
```

The data windows are relative to the current date, so results change from day to day.
