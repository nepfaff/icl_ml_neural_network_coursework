import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    neg_mean_absolute_error,
    neg_mean_squared_error,
    neg_root_mean_squared_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_absolute_percentage_error,
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
        """

        assert (
            neurons is None and activations is None or len(neurons) == len(activations)
        ), "neurons and activations lists have different lengths"

        super(Regressor, self).__init__()

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate

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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # For testing other parts until this is implemented
        return (
            torch.tensor(x).float(),
            (torch.tensor(y).float() if y is not None else None),
        )

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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
        training_data = TensorDataset(X, Y)
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
            predictions = self(X)
        return predictions

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        _, Y_true = self._preprocessor(x, y=y, training=False)  # Do not forget

        # Predict output, this function preprocess data itself so use raw data as argument
        Y_pred = self.predict(x)

        # Evaluating metrics
        evaluated = {
            "explained_variance_score": explained_variance_score(Y_true, Y_pred),
            "max_error": max_error(Y_true, Y_pred),
            "neg_mean_absolute_error": neg_mean_absolute_error(Y_true, Y_pred),
            "neg_mean_squared_error": neg_mean_squared_error(Y_true, Y_pred),
            "neg_root_mean_squared_error": neg_root_mean_squared_error(Y_true, Y_pred),
            "mean_squared_error": mean_squared_error(Y_true, Y_pred),
            "mean_squared_log_error": mean_squared_log_error(Y_true, Y_pred),
            "median_absolute_error": median_absolute_error(Y_true, Y_pred),
            "r2_score": r2_score(Y_true, Y_pred),
            "mean_poisson_deviance": mean_poisson_deviance(Y_true, Y_pred),
            "mean_gamma_deviance": mean_gamma_deviance(Y_true, Y_pred),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(
                Y_true, Y_pred
            ),
        }

        # TODO: Either to print results or return:
        # Print
        print(evaluated)

        # Return
        return evaluated

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
