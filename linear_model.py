#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*

"""

"""

import sys
import timeit
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

# Helper
def num_lasso_nonzero_weights(lassos: Tuple[Lasso]) -> float:
    """
    Trains given model and calculate testing root mean squared error.

    Parameters
    ----------
    lassos : tuple of Lasso objects
        Cross validation lasso objects.

    Return
    ------
    float
        The mean number of nonzero weights.
    """
    mean_nonzero_weights = []
    for lasso in lassos:
        mean_nonzero_weights.append(len(np.nonzero(lasso.coef_)[0]))

    return np.mean(mean_nonzero_weights)


# Scikit-learn deprecation warnings...
warnings.filterwarnings("ignore", category = DeprecationWarning)

# Pass file paths as arguments to this script and load into dataframes.
# linear_model.py [gt_pt_lm.csv]
dataset_file_path = sys.argv[1]
model_iters = float(sys.argv[2])
modeling_rounds = int(sys.argv[3])
data = pd.read_csv(dataset_file_path, index_col = [0]) # sample id as indices
print("Loading complete.")

# Initiate arrays to capture values.
# RMSE_50, RMSE_100, RMSE_150, RMSE_200, runtime
ols_vals = [[], [], [], [], []]
lasso_vals = [[], [], [], [], []]
lasso_nonzero_weights = [[], [], [], []]
ridge_vals = [[], [], [], [], []]

# Format output dataframes.
output_df = pd.DataFrame(
    data = 0.0, index = ["OLS", "Lasso", "Lasso_Non_Zero_Weights", "Ridge"],
    columns = [
        "Average_RMSE_50", "Average_RMSE_100", "Average_RMSE_150",
        "Average_RMSE_200", "Average_Runtime"
    ]
)
print("Output formatting complete.")

# Split 200 snps (features) into 4 sets, incrementing by 50.
snps = data.columns[1:].values
top_features = np.array([50, 100, 150, 200])
cv_scoring = "neg_mean_squared_error"

# Real stuff
print("Machine learning in progress...")
for i in range(modeling_rounds):
    for j in range(4):
        num_feat = top_features[j]
        # print(f"round {i} {num_feat} snps")
        curr_snps = snps[:num_feat]

        # OLS
        ols_start = timeit.default_timer()
        # print(f"ols {num_feat} snps")
        ols = LinearRegression()
        ols_cv = cross_validate(
            estimator = ols,
            X = data.loc[:, curr_snps],
            y = data.pt,
            cv = 5,
            scoring = cv_scoring,
            return_train_score = False
        )
        ols_vals[j].append(np.sqrt(-np.mean(ols_cv["test_score"])))
        ols_vals[4].append(timeit.default_timer() - ols_start)

        # Lasso
        lasso_start = timeit.default_timer()
        # print(f"lasso {num_feat} snps")
        lasso = LassoCV(cv = 5, max_iter = model_iters)
        lasso_cv = cross_validate(
            estimator = lasso,
            X = data.loc[:, curr_snps],
            y = data.pt,
            cv = 5,
            scoring = cv_scoring,
            return_train_score = False,
            return_estimator = True
        )
        lasso_vals[j].append(np.sqrt(-np.mean(lasso_cv["test_score"])))
        lasso_nonzero_weights[j].append(
            num_lasso_nonzero_weights(lasso_cv["estimator"])
        )
        lasso_vals[4].append(timeit.default_timer() - lasso_start)

        # Ridge regression
        ridge_start = timeit.default_timer()
        # print(f"ridge {num_feat} snps")
        ridge = RidgeCV(cv = 5)
        ridge_cv = cross_validate(
            estimator = ridge,
            X = data.loc[:, curr_snps],
            y = data.pt,
            cv = 5,
            scoring = cv_scoring,
            return_train_score = False
        )
        ridge_vals[j].append(np.sqrt(-np.mean(ridge_cv["test_score"])))
        ridge_vals[4].append(timeit.default_timer() - ridge_start)

# Inputting values
print("Inputting values...")
output_df["Average_RMSE_50"] = np.array([
    np.mean(ols_vals[0]), np.mean(lasso_vals[0]),
    np.mean(lasso_nonzero_weights[0]), np.mean(ridge_vals[0])
])
output_df["Average_RMSE_100"] = np.array([
    np.mean(ols_vals[1]), np.mean(lasso_vals[1]),
    np.mean(lasso_nonzero_weights[1]), np.mean(ridge_vals[1])
])
output_df["Average_RMSE_150"] = np.array([
    np.mean(ols_vals[2]), np.mean(lasso_vals[2]),
    np.mean(lasso_nonzero_weights[2]), np.mean(ridge_vals[2])
])
output_df["Average_RMSE_200"] = np.array([
    np.mean(ols_vals[3]), np.mean(lasso_vals[3]),
    np.mean(lasso_nonzero_weights[3]), np.mean(ridge_vals[3])
])
output_df["Average_Runtime"] = np.array([
    np.mean(ols_vals[4]), np.mean(lasso_vals[4]), "N/A", np.mean(ridge_vals[4])
])
output_df.to_csv("lm_comparison.csv")
print("Done!")
