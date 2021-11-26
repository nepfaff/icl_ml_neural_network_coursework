# Introduction to Machine Learning - Coursework 2 Neural Networks

## Part 2

A complete example of training, evaluating, and saving a model using defined hyperparameters is given in `part2_house_value_regression.py:main`. Furthermore, detailed docstrings are provided for all main functionality. The report provides further insight into the implementation choices.

### Training a model using given hyperparameters

A minimal version with only the required arguments and a maximal version with all possible parameters are shown below. See the corresponding docstrings for detailed information regarding every parameter and their effect.

#### Minimal version
```python
regressor = Regressor(x_train)
regressor.fit(x_train, y_train)
```

#### Maximal version
```python
regressor = Regressor(
    x_train,
    nb_epoch=500,
    batch_size=2000,
    neurons=[100, 65, 30],
    activations=["relu", "relu", "relu"],
    shuffle=True,
    learning_rate=1e-3,
    optimizer_type="sgd",
    NaN_remove_rows=False,
    NaN_mean_of_columns=True,
    NaN_fill_with_0=False,
    standardization_or_MinMax=True,
)
regressor.fit(
    x_train,
    y_train,
    log=True, 
    number_of_logs=5,
)
```

### Evaluating a trained model

A minimal version with only the required arguments and a maximal version with all possible parameters are shown below. See the corresponding docstrings for detailed information regarding every parameter and their effect.
The score method returns the MSE value. This is useful when wanting to tune hyperparameters programatically. Additional evaluation metrics are printed if `print_result` is true. These additional metrics are useful for evaluating the final model.

#### Minimal version
```python
regressor.score(
    x_test,
    y_test,
)
```

#### Maximal version
```python
regressor.score(
    x_test,
    y_test,
    print_result=True
)
```

### Tuning hyperparameters

The function `part2_house_value_regression.py:RegressorHyperParameterSearch` can be called without arguments for tuning the hyperparameters. The logic behind this is described in the report. This function is meant to be used interactively rather than just being called once. This means that one should comment out all tuning sections apart from one section. The results of the previous section should then be used for tuning the following sections. The ordering of sections in `part2_house_value_regression.py:RegressorHyperParameterSearch` corresponds to the order used for obtaining the final set of hyperparameters. See the docstring for further information.

### Tests

Unit tests where added when possible and deamed useful. These can be run from the main directory using `pytest-3`.

Further tests where added as Jupyter notebooks. These are not unit tests but more complete tests.