import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from preprocessing import RANDOM_SEED, FIG_DIR


def train_naive_bayes(train_bow, y_train):
    # Feature selection is part of the pipeline,
    # so in each fold the top-k terms are chosen using only that fold's training data.
    nb_pipe = make_pipeline(SelectKBest(chi2), MultinomialNB())

    #Values of k to try ("all" = use every feature)
    k_values = [25, 50, 100, 250, 500, 750, 1000, "all"]

    nb_grid = GridSearchCV(
        nb_pipe,
        {"selectkbest__k": k_values},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="accuracy",
    )
    nb_grid.fit(train_bow, y_train)

    nb_results_df = pd.DataFrame({
        "k": k_values,
        "mean_cv_accuracy": nb_grid.cv_results_["mean_test_score"],
        "std_cv_accuracy": nb_grid.cv_results_["std_test_score"],
    })
    print(nb_results_df.round(4))
    print("best k:", nb_grid.best_params_["selectkbest__k"], " CV accuracy:", round(nb_grid.best_score_, 4))

    nb_model = nb_grid.best_estimator_  #already refit on the full training set with the best k
    return nb_grid, nb_results_df, nb_model


def plot_naive_bayes_tuning(nb_grid, nb_results_df, train_bow):
    n_features = train_bow.shape[1]

    # Replace "all" with the actual number of features so k can be plotted on a numeric axis
    k_plot = nb_results_df["k"].replace("all", n_features).astype(int)
    best_k = nb_grid.best_params_["selectkbest__k"]
    best_k_plot = n_features if best_k == "all" else best_k

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(k_plot, nb_results_df["mean_cv_accuracy"], yerr=nb_results_df["std_cv_accuracy"], marker="o", capsize=3)
    ax.axvline(best_k_plot, ls="--", c="r", label="best k")
    ax.set_xscale("log")
    ax.set_xlabel("k (number of selected terms)")
    ax.set_ylabel("CV accuracy")
    ax.set_title("Multinomial naive Bayes: top-k feature selection")
    ax.legend()
    plt.savefig(f"{FIG_DIR}/nb_tuning.png", dpi=200, bbox_inches="tight")
    plt.close()


def train_logistic_regression(train_bow, y_train, test_bow, y_test):
    lrmodel = LogisticRegressionCV(Cs = np.logspace(-3, 3, 30), solver = 'liblinear', penalty= "l1", cv = 5, random_state=RANDOM_SEED)
    lrmodel.fit(train_bow, y_train)
    print(lrmodel.score(test_bow, y_test))
    return lrmodel


def plot_logistic_regression_tuning(lrmodel):
    s = lrmodel.scores_[lrmodel.classes_[1]]
    x = lrmodel.Cs_
    mean, std = s.mean(axis=0), s.std(axis=0)
    print(lrmodel.C_)

    plt.errorbar(x, mean, yerr=std, marker="o", capsize=3)
    plt.axvline(lrmodel.C_[0], ls="--", c="r", label="best")
    plt.xscale("log")
    plt.xlabel("C"); plt.ylabel("CV accuracy"); plt.legend()
    plt.savefig(f"{FIG_DIR}/lr_tuning.png", dpi=200, bbox_inches="tight")
    plt.close()


def train_tree(train_bow, y_train, test_bow, y_test):
    path = DecisionTreeClassifier(random_state=RANDOM_SEED).cost_complexity_pruning_path(train_bow, y_train)
    alphas = path.ccp_alphas[:-1]

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        {"ccp_alpha": alphas},
        cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_SEED),
        scoring="accuracy",
    )
    grid.fit(train_bow, y_train)
    print('best score: ', grid.best_score_, ' best params: ', grid.best_params_)
    print(classification_report(y_test, grid.predict(test_bow), digits=3))
    return grid


def plot_tree_tuning(grid):
    res = grid.cv_results_
    x = res["param_ccp_alpha"].data.astype(float)
    mean, std = res["mean_test_score"], res["std_test_score"]

    plt.errorbar(x, mean, yerr=std, marker="o", capsize=3)
    plt.axvline(grid.best_params_["ccp_alpha"], ls="--", c="r", label="best")
    plt.xscale("log")
    plt.xlabel("ccp_alpha"); plt.ylabel("CV accuracy"); plt.legend()
    plt.savefig(f"{FIG_DIR}/tree_tuning.png", dpi=200, bbox_inches="tight")
    plt.close()


def tune_random_forest(train_bow, y_train):
    performance = []

    # Random Forests only accept numeric input, so we use the bag-of-words representation
    for max_features in [None, "sqrt", "log2"]:
        for n_estimators in [100, 500, 1000]:
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED, max_features=max_features, oob_score=True)
            rf_model.fit(train_bow, y_train)
            performance.append({"n_estimators": n_estimators,
                "max_features": str(max_features),
                "oob_accuracy": rf_model.oob_score_,})

    rf_results_df = pd.DataFrame(performance)
    return rf_results_df


def train_random_forest(train_bow, y_train):
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=RANDOM_SEED, max_features="log2", oob_score=True)
    rf_model.fit(train_bow, y_train)
    return rf_model


def plot_random_forest_tuning(rf_results_df):
    colors = {"None": "blue", "sqrt": "orange", "log2": "green"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for setting, group in rf_results_df.groupby("max_features"):
        ax.scatter(
            group["n_estimators"],
            group["oob_accuracy"],
            color=colors[setting],
            label=setting,
            s=80,
        )

    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB accuracy")
    ax.legend(title="max_features", bbox_to_anchor=(1, 1))
    plt.savefig(f"{FIG_DIR}/rf_tuning.png", dpi=200, bbox_inches="tight")
    plt.close()


def tune_gradient_boosting(train_bow, y_train):
    param_grid = {
        "n_estimators": [100, 500, 1000],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [1, 3, 5]
    }

    gbc = GradientBoostingClassifier(random_state=RANDOM_SEED)
    grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)   # n_jobs=-1: use all CPU cores
    grid_search.fit(train_bow, y_train)

    grad_results_df = pd.DataFrame(grid_search.cv_results_)[["param_n_estimators", "param_learning_rate", "param_max_depth", "mean_test_score"]]
    return grid_search, grad_results_df


def plot_gradient_boosting_tuning(grad_results_df):
    colors = {0.01: "blue", 0.1: "orange", 0.5: "green"}
    markers = {1: "o", 3: "s", 5: "^"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for (lr, depth), group in grad_results_df.groupby(["param_learning_rate", "param_max_depth"]):
        ax.scatter(
            group["param_n_estimators"],
            group["mean_test_score"],
            color=colors[lr],
            marker=markers[depth],
            label=f"lr={lr}, depth={depth}",
            s=70,
        )

    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gradient boosting hyperparameter tuning")
    ax.legend(title="Learning rate and depth", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    plt.savefig(f"{FIG_DIR}/gb_tuning.png", dpi=200, bbox_inches="tight")
    plt.close()
