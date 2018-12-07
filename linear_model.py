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
from sklearn.model_selection import cross_validate, train_test_split


# Helpers
def test_rmse(
        models: Tuple,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
    """
    Trains given model and calculate testing root mean squared error.

    Parameters
    ----------
    models : Tuple
        A tuple of cross-validation split estimators

    X_test : pd.DataFrame
        The testing set of SNPs.

    y_test : pd.Series
        The testing set of phenotype values.

    Return
    ------
    float
        The root mean squared error.
    """
    rmse = 0
    for model in models:
        y_pred = model.predict(X_test)
        rmse += np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse / len(models)


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

# Format output dataframes.
output_df = pd.DataFrame(
    data = 0.0, index = ["OLS", "Lasso", "Ridge"],
    columns = [
        "Average_RMSE_50", "Average_R2_50",
        "Average_RMSE_100", "Average_R2_100",
        "Average_RMSE_150", "Average_R2_150",
        "Average_RMSE_200", "Average_R2_200", "Average_Runtime"
    ]
)
lasso_weights_df = pd.DataFrame(
    data = 0.0, index = ["Lasso_Non_Zero_Weights"],
    columns = ["Average_50", "Average_100", "Average_150", "Average_200"]
)
print("Output formatting complete.")

# Split 200 snps (features) into 4 sets, incrementing by 50.
snps = data.columns[1:].values
top_features = np.array([50, 100, 150, 200])
cv_scoring = ("r2", "neg_mean_squared_error")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, snps], data.pt, test_size = 30
)

# Real stuff
print("Machine learning in progress...")
for i in range(modeling_rounds):
    print(f"Round {i + 1}...")
    for j in range(4):
        num_feat = top_features[j]
        curr_snps = snps[:num_feat]

        # OLS
        ols_start = timeit.default_timer()
        ols = LinearRegression()
        ols_cv = cross_validate(
            estimator = ols,
            X = X_train.loc[:, curr_snps],
            y = y_train,
            cv = 5,
            return_estimator = True
        )
        output_df.at["OLS", str("Average_RMSE_" + str(num_feat))] += \
            test_rmse(ols_cv["estimator"], X_test.loc[:, curr_snps], y_test)
        output_df.at["OLS", str("Average_R2_" + str(num_feat))] += \
            np.mean(ols_cv["test_score"], )
        output_df.at["OLS", "Average_Runtime"] += \
            (timeit.default_timer() - ols_start)

        # Lasso
        lasso_start = timeit.default_timer()
        lasso = LassoCV(cv = 5, max_iter = model_iters)
        lasso_cv = cross_validate(
            estimator = lasso,
            X = X_train.loc[:, curr_snps],
            y = y_train,
            cv = 5,
            return_estimator = True
        )
        output_df.at["Lasso", str("Average_RMSE_" + str(num_feat))] += \
            test_rmse(lasso_cv["estimator"], X_test.loc[:, curr_snps], y_test)
        output_df.at["Lasso", str("Average_R2_" + str(num_feat))] += \
            np.mean(lasso_cv["test_score"])
        lasso_weights_df.at[
            "Lasso_Non_Zero_Weights", str("Average_" + str(num_feat))
        ] += num_lasso_nonzero_weights(lasso_cv["estimator"])
        output_df.at["Lasso", "Average_Runtime"] += \
            (timeit.default_timer() - lasso_start)

        # Ridge regression
        ridge_start = timeit.default_timer()
        ridge = RidgeCV(cv = 5)
        ridge_cv = cross_validate(
            estimator = ridge,
            X = X_train.loc[:, curr_snps],
            y = y_train,
            cv = 5,
            return_estimator = True
        )
        output_df.at["Ridge", str("Average_RMSE_" + str(num_feat))] += \
            test_rmse(ridge_cv["estimator"], X_test.loc[:, curr_snps], y_test)
        output_df.at["Ridge", str("Average_R2_" + str(num_feat))] += \
            np.mean(ridge_cv["test_score"])
        output_df.at["Ridge", "Average_Runtime"] += \
            (timeit.default_timer() - ridge_start)


output_df /= modeling_rounds
lasso_weights_df /= modeling_rounds
output_df.to_csv("lm_comparison.csv")
lasso_weights_df.to_csv("lasso_nonzero_weights.csv")
print("Done!")
