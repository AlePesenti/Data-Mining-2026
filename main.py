from preprocessing import (
    load_data, missing_text_per_class, most_frequent_texts, build_text_raw, review_length, surface_cues,
    clean_text, get_model_inputs, split_data, bag_of_words,
)
from train_models import (
    train_naive_bayes, plot_naive_bayes_tuning,
    train_logistic_regression, plot_logistic_regression_tuning,
    train_tree, plot_tree_tuning,
    tune_random_forest, train_random_forest, plot_random_forest_tuning,
    tune_gradient_boosting, plot_gradient_boosting_tuning,
)
from evaluate import evaluate_models, question_1, question_2, question_3, question_4


def main():
    # Data and pre-processing
    df = load_data()
    missing_text_per_class(df)
    most_frequent_texts(df)
    df = build_text_raw(df)
    df = review_length(df)
    surface_cues(df)
    df = clean_text(df)
    X_text, y, sentiment = get_model_inputs(df)
    X_train, X_test, y_train, y_test, sentiment_train, sentiment_test = split_data(X_text, y, sentiment)
    vectorizer, train_bow, test_bow = bag_of_words(X_train, X_test)

    # Classifiers and hyperparameter tuning
    nb_grid, nb_results_df, nb_model = train_naive_bayes(train_bow, y_train)
    plot_naive_bayes_tuning(nb_grid, nb_results_df, train_bow)

    lrmodel = train_logistic_regression(train_bow, y_train, test_bow, y_test)
    plot_logistic_regression_tuning(lrmodel)

    grid = train_tree(train_bow, y_train, test_bow, y_test)
    plot_tree_tuning(grid)

    rf_results_df = tune_random_forest(train_bow, y_train)
    rf_model = train_random_forest(train_bow, y_train)
    plot_random_forest_tuning(rf_results_df)

    grid_search, grad_results_df = tune_gradient_boosting(train_bow, y_train)
    plot_gradient_boosting_tuning(grad_results_df)

    # Evaluation on the test set
    best_rf, best_gbc = evaluate_models(train_bow, y_train, test_bow, y_test)

    # Questions
    question_1(nb_model, lrmodel, test_bow, y_test)
    question_2(grid, rf_model, grid_search, test_bow, y_test)
    question_3(best_rf, test_bow, y_test, sentiment_test)
    question_4(vectorizer, lrmodel, grid, best_rf, best_gbc)


if __name__ == "__main__":
    main()
