import numpy as np
import pandas as pd
import yfinance as yf
import itertools
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
from initialize import *
from baum_welch import *
import matplotlib.pyplot as plt

input = ["BTC-USD", "ETH-USD"]
iteration_number=1000
optimums = []

datas = load_data(input, 90, 30)
merged_states = calculate_states(datas, 3) # Calculate states for Bitcoin and Ethereum with 3 states

observations = calculate_states(load_data(input, 30, 0), 3)["Change"].values.tolist()

result = baum_welch_algorithm(
    observations,
    merged_states,
    3,
    len(input),
    True,
    110
)

print(result)

for _ in range(iteration_number):
    a, pi, means, covs, b, opt = baum_welch_algorithm(
    observations,
    merged_states,
    3,
    len(input),
    False,
    300
    )
    optimums.append(opt)

print(max(optimums))

