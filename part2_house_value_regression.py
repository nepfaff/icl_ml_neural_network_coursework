import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math
from statistics import mean
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
)


class Regressor(nn.Module):
    def __init__(
        self,
        x,
        nb_epoch=1000,
        neurons=None,
        activations=None,
        batch_size=100,
        shuffle=True,
        learning_rate=1e-3,
        optimizer_type="sgd",
        NaN_remove_rows=False,
        NaN_mean_of_columns=False,
        NaN_fill_with_0=True,
        standardization_or_MinMax=True,
    ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers. This excludes the input and output layer.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer. Must have the same length as 'neurons'
                The first element is used as the activation for the input layer.
                Allowed options are "relu", "sigmoid", "tanh". Anything else is ignored;
                e.g. "identity" or "linear".
            - batch_size {int} -- Batch size for mini batch gradient descent.
            - shuffle {bool} -- Shuffle the input data before training.
            - learning_rate {float} -- Learning rate for backward pass. Only used when
                'optimizer_type' is "sgd".
            - optimizer_type {str} -- The optimizer type to use. One of "sgd",
                "adadelta", or "adam".
            - NaN_remove_rows {bool} -- Remove rows containing NaN in preprocessor method
            - NaN_mean_of_columns {bool} -- Replace NaN with mean of respective coulmun in preprocessor method
            - NaN_fill_with_0 {bool} -- Replace NaN with 0 in preprocessor method
            - standardization_or_MinMax {bool} -- If true perfrom standardization when preprocessing data.
                If false perform MinMax normalization when preprocessing data.
        """

        assert (
            neurons is None and activations is None or len(neurons) == len(activations)
        ), "neurons and activations lists have different lengths"

        super(Regressor, self).__init__()

        # Divide and multiply y-values with this constant to improve learning
        self._y_scale = 100000

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate

        # Preprocessor parameters
        self.NaN_remove_rows = NaN_remove_rows
        self.NaN_mean_of_columns = NaN_mean_of_columns
        self.NaN_fill_with_0 = NaN_fill_with_0

        self.standardization_or_MinMax = standardization_or_MinMax

        # Default values
        neurons = [10, 10] if neurons is None else neurons
        activations = ["relu", "relu"] if activations is None else activations

        # Determine input and output layer sizes
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1

        # Construct network structure
        neurons = [self.input_size, *neurons, self.output_size]
        activations.append("identity")  # output layer
        layers = []
        for i in range(len(neurons) - 1):
            # Linear layer
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            # Activation
            if activations[i] == "relu":
                layers.append(nn.ReLU())
            elif activations[i] == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activations[i] == "tanh":
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Optimizer
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer_type == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.parameters())
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters())
        else:
            raise AttributeError(f"Invalid optimizer type: {optimizer_type}")

    def forward(self, X):
        """
        PyTorch forward method

        Arguments:
            - X {torch.Tensor} -- Input tensor of shape (batch_size, input_size).

        Returns:
            {torch.Tensor} -- Predicted value for the given input (batch_size, 1).
        """

        return self.layers(X)

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.


        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
        """

        # Handle NaN values
        if self.NaN_remove_rows:
            # Remove rows with NAN
            x = x.dropna()
            if y is not None:
                y = y.dropna()

        if self.NaN_mean_of_columns:
            # Replace NaN with mean of  respective coulmun
            x = x.fillna(x.mean())
            if y is not None:
                y = y.fillna(y.mean())

        if self.NaN_fill_with_0:
            # Replace NaN with 0
            x = x.fillna(0)
            if y is not None:
                y = y.fillna(0)

        # Convert inputs to np.ndarray
        x = x.values
        if y is not None:
            y = y.values

            # Scale y-values
            y /= self._y_scale

        if training:
            # Handle textual values in the data, encoding them using one-hot encoding
            lb = preprocessing.LabelBinarizer()
            x = np.concatenate((x[:, :-1], lb.fit_transform(x[:, -1])), axis=1)

            # Store Binarizer preprocessing parameters
            self.lb_training = lb

            if self.standardization_or_MinMax:
                # Perform Standardization
                ss = preprocessing.StandardScaler()
                x = ss.fit_transform(x)
                # y = ss.fit_transform(y)

                # Store Standardization preprocessing parameters
                self.ss_training = ss
            else:
                # Perform MinMax Normalization
                mms = preprocessing.MinMaxScaler()
                x = mms.fit_transform(x)

                # Store mms preprocessing parameters
                self.mms_training = mms
        else:
            # Handle textual values in the data, encoding them using one-hot encoding
            x = np.concatenate(
                (x[:, :-1], self.lb_training.transform(x[:, -1])), axis=1
            )

            if self.standardization_or_MinMax:
                # Perform Standardization
                x = self.ss_training.transform(x)
            else:
                # Perform MinMax Normalization
                x = self.mms_training.transform(x)

        # Return preprocessed x and y, return None for y if it was None
        return x.astype(float), (y.astype(float) if y is not None else None)

    def fit(self, x, y, log=False, number_of_logs=5):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - log {bool} -- Log gradient descent loss improvements.
            - number_of_logs {int} -- Number of log messages to display.

        Returns:
            self {Regressor} -- Trained model.

        """

        # Create data loader
        X, Y = self._preprocessor(x, y=y, training=True)
        training_data = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        training_data_loader = DataLoader(
            training_data, batch_size=self.batch_size, shuffle=self.shuffle
        )

        # Training loop
        for epoch in range(self.nb_epoch):
            # Perform mini batch gradient descent
            for X, y in training_data_loader:
                # Forward pass
                predictions = self(X)
                loss = self.loss_fn(predictions, y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Optional logging
            if log and epoch % (self.nb_epoch // number_of_logs) == 0:
                print(f"epoch: {epoch}, loss (based on last batch): {loss}")

        return self

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)
        with torch.no_grad():
            scaled_predictions = self(torch.Tensor(X)).detach().numpy()

        # Invert y-value scaling form preprocessor
        predictions = scaled_predictions * self._y_scale

        return predictions

    def score(self, x, y, print_result=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        _, Y_true = self._preprocessor(x, y=y, training=False)  # Do not forget

        # Predict output, this function preprocess data itself so use raw data as argument
        Y_pred = self.predict(x)

        assert type(Y_true) == type(
            Y_pred
        ), "y_true and y_pred are different types, both should be nd.array"
        assert len(Y_true) == len(Y_pred) and len(Y_true[0]) == len(
            Y_pred[0]
        ), "y_true and y_pred are different shapes, both should be (batch_size, 1)"

        # Evaluating metrics
        evaluated = {
            # "explained_variance_score": explained_variance_score(Y_true, Y_pred),
            "mean_squared_error": mean_squared_error(Y_true, Y_pred),
            "median_absolute_error": median_absolute_error(Y_true, Y_pred),
            # "r2_score": r2_score(Y_true, Y_pred),
            # "mean_poisson_deviance": mean_poisson_deviance(Y_true, Y_pred),
            # "mean_gamma_deviance": mean_gamma_deviance(Y_true, Y_pred),
        }

        if print_result:
            print(evaluated)

        return evaluated["mean_squared_error"]


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def generate_neurons_in_hidden_layers(
    n_hidden_layers, n_neurons_first_hidden_layer, n_neurons_last_hidden_layer
):
    """
    Generate number of neurons in hidden layers for hyperparameter tuning.

    :param n_hidden_layers: Number of hidden layers.
    :param n_neurons_first_hidden_layer: Number of neurons in the first hidden layer.
    :param n_neurons_last_hidden_layer: Number of neurons in the last hidden layer.
    """

    if n_hidden_layers == 0:
        return []
    elif n_hidden_layers == 1:
        return [n_neurons_first_hidden_layer]

    n_neurons_in_hidden_layer = []
    n_neurons_increment = (
        n_neurons_last_hidden_layer - n_neurons_first_hidden_layer
    ) / (n_hidden_layers - 1)
    n_neurons = n_neurons_first_hidden_layer
    for _ in range(n_hidden_layers):
        n_neurons_in_hidden_layer.append(math.ceil(n_neurons))
        n_neurons += n_neurons_increment

    return n_neurons_in_hidden_layer


def j_fold_split(n_instances: int, j: int = 3, random_generator=default_rng()):
    """
    Randomises indices and splits them into j folds

    :param n_instances: Number of instances of the dataset.
    :param j: Number of folds for splitting.
    :param random_generator: A random generator (np.random.Generator).
    :return: A list of length j. Each element in the list is a numpy array of
        shape (n,) and type int, giving the indices of the instances in that fold.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, j)

    return split_indices


def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    # Use pandas to read CSV data as it contains various object types
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"

    # Spliting input and output
    x = data.loc[:, data.columns != output_label].values
    y = data.loc[:, [output_label]].values

    # Training
    best_error = float("inf")
    best_layers = -1
    best_n_neurons_first_hidden_layer = -1
    n_neurons_last_hidden_layer = -1
    for n_layers in [1, 3, 5, 8, 10]:
        for n_neurons_first_hidden_layer in [10, 30, 60, 100]:
            for n_neurons_last_hidden_layer in [10, 30, 60, 100]:
                # Cross-validation
                errors = []

                # Randomise data & split code into j folds
                split_indices = j_fold_split(len(x), 3)
                for i, fold in enumerate(split_indices):
                    # Assign test and train data
                    test_indices = fold
                    train_indices = np.hstack(
                        split_indices[:i] + split_indices[i + 1 :]
                    )
                    x_train = pd.DataFrame(x[train_indices])
                    y_train = pd.DataFrame(y[train_indices])
                    x_test = pd.DataFrame(x[test_indices])
                    y_test = pd.DataFrame(y[test_indices])

                    neurons = generate_neurons_in_hidden_layers(
                        n_layers,
                        n_neurons_first_hidden_layer,
                        n_neurons_last_hidden_layer,
                    )
                    activations = ["relu" for _ in range(len(neurons))]
                    regressor = Regressor(
                        x_train,
                        nb_epoch=500,
                        batch_size=2000,
                        neurons=neurons,
                        activations=activations,
                        optimizer_type="adadelta",
                    )
                    regressor.fit(x_train, y_train)

                    error = regressor.score(x_test, y_test)
                    errors.append(error)

                # Error
                mean_error = mean(errors)
                print(
                    f"Regressor error: {mean_error},"
                    + f" layers: {n_layers},"
                    + f" n_neurons_first_hidden_layer: {n_neurons_first_hidden_layer},"
                    + f" n_neurons_last_hidden_layer: {n_neurons_last_hidden_layer}"
                )

                if mean_error < best_error:
                    best_error = mean_error
                    best_layers = n_layers
                    best_n_neurons_first_hidden_layer = n_neurons_first_hidden_layer
                    best_n_neurons_last_hidden_layer = n_neurons_last_hidden_layer

    print(
        f"\nBest overall -> layers: {best_layers},"
        + f" n_neurons_first_hidden_layer: {best_n_neurons_first_hidden_layer},"
        + f" n_neurons_last_hidden_layer: {best_n_neurons_last_hidden_layer}"
    )


def example_main():
    # Use pandas to read CSV data as it contains various object types
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Splitting into training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Training
    neurons = [30, 15]
    activations = ["relu", "relu"]
    regressor = Regressor(
        x_train,
        nb_epoch=500,
        batch_size=2000,
        learning_rate=1e-2,
        neurons=neurons,
        activations=activations,
        optimizer_type="sgd",
    )
    regressor.fit(x_train, y_train, log=True, number_of_logs=10)

    # Save model
    # save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # example_main()
    RegressorHyperParameterSearch()
