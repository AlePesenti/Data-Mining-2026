"""Evaluation on the test set and answers to the assignment questions.

Predictive performance is measured with accuracy, precision, recall and the F1 score,
with fake (AI = 1) as the positive class.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

from preprocessing import RANDOM_SEED


def evaluate_models(train_bow, y_train, test_bow, y_test):
    """Quantitative evaluation."""
    # Multinomial Naive Bayes Evaluation

    # Logistic Regression Evaluation

    # Single Classification Tree Evaluation

    # Random Forests Evaluation
    best_rf = RandomForestClassifier(n_estimators=1000, max_features="log2", random_state=RANDOM_SEED)
    best_rf.fit(train_bow, y_train)
    y_pred = best_rf.predict(test_bow)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("Random Forest Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("\n")

    # Gradient Boosting Evaluation
    best_gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5, max_depth=1, random_state=RANDOM_SEED)
    best_gbc.fit(train_bow, y_train)
    y_pred = best_gbc.predict(test_bow)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("Gradient Boosting Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    return best_rf, best_gbc


def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):
    """McNemar's test (Dietterich, 1998) on the difference in accuracy between two classifiers, alpha = 0.01."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = np.array([[np.sum(a_ok & b_ok),  np.sum(a_ok & ~b_ok)],
                      [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]])
    exact = table[0, 1] + table[1, 0] < 25
    res = mcnemar(table, exact=exact, correction=True)
    print(name_a, " vs ",name_b, "\n", table)
    print("statistic=",res.statistic, "p=", res.pvalue,
          {'significant' if res.pvalue < 0.01 else 'not significant'}, " at α=0.01\n")


def question_1(nb_model, lrmodel, test_bow, y_test):
    """How does the performance of the generative linear model (multinomial naive Bayes) compare to
    the discriminative linear model (regularized logistic regression)?
    """
    # Final models
    nb_pred = nb_model.predict(test_bow)
    lr_pred = lrmodel.predict(test_bow)

    # Test-set performance, positive class=1 AI)
    q1_results = pd.DataFrame({
        name: {
            "accuracy":  accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, pos_label=1),
            "recall":    recall_score(y_test, pred, pos_label=1),
            "f1":        f1_score(y_test, pred, pos_label=1),
        }
        for name, pred in [("Naive Bayes", nb_pred), ("Logistic regression (L1)", lr_pred)]
    }).T
    print(q1_results.round(3), "\n")

    #Is the difference in accuracy significant? McNemar's test (alpha = 0.01), function defined above
    mcnemar_test(np.asarray(y_test), nb_pred, lr_pred, "NB", "LR")


def question_2(grid, rf_model, grid_search, test_bow, y_test):
    """Do the tree ensembles improve on the performance of the single tree?"""
    y_test = np.asarray(y_test)
    mcnemar_test(y_test, grid.predict(test_bow), rf_model.predict(test_bow), "Tree", "RF")
    mcnemar_test(y_test, grid.predict(test_bow), grid_search.predict(test_bow), "Tree", "GB")


def question_3(best_rf, test_bow, y_test, sentiment_test):
    """For the random forest classifier finally selected, is it better at distinguishing between fake and
    genuine positive reviews or between fake and genuine negative reviews?

    A statistical test determines whether the difference in accuracy between positive and negative reviews
    is significant.
    """
    rf = best_rf.predict(test_bow)

    positive_reviews = sentiment_test == "POS"
    negative_reviews = sentiment_test == "NEG"

    correctly_predicted = rf == y_test

    table = pd.DataFrame([
        [np.sum(correctly_predicted & positive_reviews), np.sum(correctly_predicted & negative_reviews)],
        [np.sum(~correctly_predicted & positive_reviews), np.sum(~correctly_predicted & negative_reviews)]
    ], index=['Correctly Predicted', 'Incorrectly Predicted'], columns=['Positive', 'Negative'])

    print(table)

    accuracies = table.loc["Correctly Predicted"] / table.sum()

    pos_accuracy = accuracies["Positive"]
    neg_accuracy = accuracies["Negative"]

    print(f"Positive review accuracy: {pos_accuracy}")
    print(f"Negative review accuracy: {neg_accuracy}")
    print(f"Difference: {100 * (pos_accuracy - neg_accuracy)}")


    chi2, p_value, dof, expected = chi2_contingency(
        table.to_numpy(),
        correction=False
    )

    print(f"p-value: {p_value}")
    print("Expected counts:", expected)


def top_features_logistic_regression(lrmodel, feature_names):
    coefs = pd.Series(lrmodel.coef_[0], index=feature_names)

    print("Non null terms:", (coefs != 0).sum(), "over", len(coefs))

    top_ai    = coefs.sort_values(ascending=False).head(5)
    top_human = coefs.sort_values().head(5)

    print("\nTop 5 -> AI\n", top_ai.round(3))
    print("\nTop 5 -> Human\n", top_human.round(3))


def top_features_tree(grid, feature_names):
    tree_model = grid.best_estimator_
    tree_imp = pd.Series(tree_model.feature_importances_, index=feature_names)

    print("Feature used by the tree:", (tree_imp > 0).sum())
    print(tree_imp.sort_values(ascending=False).head(5).round(3))


def top_features_random_forest(best_rf, feature_names):
    best_rf_features = pd.Series(best_rf.feature_importances_, index=feature_names)

    # Get 5 best features
    top_rf = best_rf_features.sort_values(ascending=False).head(5)
    print("\nTop 5 -> Random Forest\n", top_rf.round(3))


def top_features_gradient_boosting(best_gbc, feature_names):
    best_gbc_features = pd.Series(best_gbc.feature_importances_, index=feature_names)

    # Get 5 best features
    top_gbc = best_gbc_features.sort_values(ascending=False).head(5)
    print("\nTop 5 -> Gradient Boosting\n", top_gbc.round(3))


def question_4(vectorizer, lrmodel, grid, best_rf, best_gbc):
    """For each algorithm, what are the 5 most important features in the final model? For the linear models,
    make a distinction between features pointing towards an AI-generated review and features pointing towards
    a review written by a human.
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features_logistic_regression(lrmodel, feature_names)
    top_features_tree(grid, feature_names)
    top_features_random_forest(best_rf, feature_names)
    top_features_gradient_boosting(best_gbc, feature_names)
