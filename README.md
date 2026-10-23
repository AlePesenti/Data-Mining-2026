# Data Mining 2026 - Fake hotel reviews (Group 33)

## Running the code

Put `all_data.csv` in a `data` folder next to the scripts, then run from the main folder:

```
python main.py
```

It takes around 8 minutes on a laptop (the gradient boosting grid search is the slow part). Everything is printed in the terminal and the plots are saved in `figures/`. The seed is fixed to 42 in `preprocessing.py`, so running it again gives the same numbers.

## Files

- `main.py` calls everything in order
- `preprocessing.py`: loading the English reviews, data exploration, text cleaning, train/test split and bag-of-words
- `train_models.py`: tuning and training of the five models + tuning plots
- `evaluate.py`: test set evaluation and questions 1 to 4

## Where the results in the report come from

- Data and pre-processing (sections 2-3): `preprocessing.py`
- Hyperparameter tuning (section 4.2 and appendix B, figures in `figures/`): `train_models.py`
- Test performance of the models (section 5.1): `evaluate_models` and `question_1` in `evaluate.py`, the tree scores are printed by `train_tree` in `train_models.py`
- Q1-Q4 (sections 5.2-5.5): the functions `question_1` ... `question_4` in `evaluate.py`

## Libraries

We used Python 3.11 with numpy, pandas, scikit-learn, scipy, statsmodels and matplotlib:

```
pip install numpy pandas scikit-learn scipy statsmodels matplotlib
```
