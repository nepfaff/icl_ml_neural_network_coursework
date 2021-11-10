import numpy as np
from numpy.random import default_rng
import pickle
import math

from numpy.lib.function_base import vectorize


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # log information needed to compute gradient at a later stage
        self._cache_current = x

        # perform forward pass through sigmoid layer. (apply sigmoid function elementwise to input)
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        # Check that grad_z has the correct shape
        assert (
            len(grad_z.shape) == 2 and grad_z.shape == self._cache_current.shape
        ), "grad_z has incorrect shape"

        # Compute derivative of sigmoid function
        sigmoid_derivative = lambda k: (1 / (1 + np.exp(-k))) * (
            1 - (1 / (1 + np.exp(-k)))
        )

        # Compute the gradient with respect to the layer inputs
        return grad_z * sigmoid_derivative(self._cache_current)


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # log information needed to compute gradient at a later stage
        self._cache_current = x

        # perform forward pass through Relu layer. (apply Relu function elementwise to input)
        x[x <= 0] = 0
        return x

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        # Check that grad_z has the correct shape
        assert (
            len(grad_z.shape) == 2 and grad_z.shape == self._cache_current.shape
        ), "grad_z has incorrect shape"

        # Compute derivative of relu function with respect to inputs of layer and sore it in relu_derivative
        relu_derivative = np.where(self._cache_current <= 0, 0, 1)

        # Compute the gradient with respect to the layer inputs
        return grad_z * relu_derivative


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """

        # Store the layer dimensions
        self.n_in = n_in
        self.n_out = n_out

        # Initialise the learnable weights and bias
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros(n_out, dtype=float)

        # Initialise the parameters required for backpropagation
        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        # Check that x has the correct shape
        assert len(x.shape) == 2 and x.shape[1] == self.n_in, "x has incorrect shape"

        # Store the data necessary for computing the gradients
        self._cache_current = x

        return np.matmul(x, self._W) + self._b

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        # Check that grad_z has the correct shape
        assert len(grad_z.shape) == 2 and grad_z.shape == (
            len(self._cache_current),
            self.n_out,
        ), "grad_z has incorrect shape"

        # Compute the gradients with respect to the layer's parameters
        self._grad_W_current = np.matmul(self._cache_current.T, grad_z)
        self._grad_b_current = np.sum(grad_z, axis=0)

        # Compute and return the gradient with respect to the layer inputs
        return np.matmul(grad_z, self._W.T)

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        # Update weights and bias using the previously calculated gradients
        self._W -= learning_rate * self._grad_W_current
        self._b -= learning_rate * self._grad_b_current


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Layer Instances of network containing alternating pattern of linear layers and activation function layers, type: list
        f_plus_n = np.append(np.array(self.input_dim), self.neurons)

        linear_layers = []

        for i, _ in enumerate(neurons):
            layer = LinearLayer(f_plus_n[i], f_plus_n[i + 1])
            linear_layers.append(layer)
            if activations[i] == "relu":
                act_layer = ReluLayer()
                linear_layers.append(act_layer)
            elif activations[i] == "sigmoid":
                act_layer = SigmoidLayer
                linear_layers.append(act_layer)

        self._layers = linear_layers
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for layer in self._layers:
            x = layer.forward(x)

        # TODO: test this!
        assert type(x) == np.ndarray

        return x  # np.zeros((1, self.neurons[-1]))  # Replace with your own code

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)

        # TODO: test this!
        assert type(grad_z) == np.ndarray

        return grad_z  # np.zeros((1, self.neurons[-1]))  # Replace with your own code

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self, network, batch_size, nb_epoch, learning_rate, loss_fun, shuffle_flag
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        self._loss_layer = (
            MSELossLayer() if loss_fun == "mse" else CrossEntropyLossLayer()
        )

    @staticmethod
    def shuffle(input_dataset, target_dataset, rg=default_rng()):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).
            - rg (numpy.random.Generator) -- Random number generator.

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """

        assert len(input_dataset) == len(
            target_dataset
        ), "input and target data sets have different lengths"

        # Shuffled indices for shuffling the instances
        shuffled_indices = rg.permutation(len(input_dataset))

        # Return the input data sets in shuffled order
        return input_dataset[shuffled_indices], target_dataset[shuffled_indices]

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """

        assert len(input_dataset) == len(
            target_dataset
        ), "input and target data sets have different lengths"

        # Determine the number of batches
        number_of_batches = len(input_dataset) // self.batch_size

        # Loop for the number of epochs
        for _ in self.nb_epoch:
            if self.shuffle_flag:
                # Shuffle the data sets
                input_dataset, target_dataset = self.shuffle(
                    input_dataset, target_dataset
                )

            # Split data sets into batches
            input_batches = np.array_split(input_dataset, number_of_batches)
            target_batches = np.array_split(target_dataset, number_of_batches)

            # Loop over all batches
            for input_batch, target_batch in zip(input_batches, target_batches):
                # Forward pass
                predicted_batch = self.network(input_batch)

                # Compute loss (necessary for backward pass to work)
                self._loss_layer.forward(
                    predicted_batch, target_batch
                )  # Return value 'loss' is ignored

                # Backward pass
                grad_loss_wrt_outputs = self._loss_layer.backward()
                self.network.backward(
                    grad_loss_wrt_outputs
                )  # Return value 'grad_loss_wrt_inputs' is ignored

                # Gradient descent on the network parameters
                self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        """

        assert len(input_dataset) == len(
            target_dataset
        ), "input and target data sets have different lengths"

        # Get the predicted values
        network_output = self.network(input_dataset)

        # Evaluate and return the loss between the predicted and actual outputs
        return self._loss_layer.forward(network_output, target_dataset)


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """

        # Scale the data to values between self.a = 0 and self.b = 1.
        self.a = 0
        self.b = 1

        # Check if data has shape (n,) and convert to (1, n)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        # Determine X_Max and X_Min for each feature (column) in the input data.
        self.X_Max = np.max(data, axis=0)
        self.X_Min = np.min(data, axis=0)

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            normalized_data.T {np.ndarray} normalized dataset.
        """

        # Check if data has shape (n,) and convert to (1, n)
        converted_shape = False
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
            converted_shape = True

        # Store columns of normalized values as rows in list data_norm
        data_norm = []

        for column in range(data.shape[1]):
            # Calculate normalised values for each column (features), however, stored as rows
            data_norm_column = self.a + (
                (data[:, column] - self.X_Min[column])
                / (self.X_Max[column] - self.X_Min[column])
            ) * (self.b - self.a)

            # Add new normalised row to data_norm
            data_norm.append(data_norm_column)

        # Convert data_norm to an array and return the transpose since it contains the normalised columns of data as rows
        normalized_data = np.array(data_norm)
        if converted_shape:
            # Convert back to (n,)
            return normalized_data.flatten()
        return normalized_data.T

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            reverted_data.T {np.ndarray} reverted dataset.
        """

        # Check if data has shape (n,) and convert to (1, n)
        converted_shape = False
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
            converted_shape = True

        # Store columns of reverted values as rows in list data_rev
        data_rev = []

        for column in range(data.shape[1]):
            # Calculate reverted values for each column (features), however, stored as rows
            data_rev_column = self.X_Min[column] + (
                (data[:, column] - self.a) / (self.b - self.a)
            ) * (self.X_Max[column] - self.X_Min[column])

            # Add new reverted row to data_rev
            data_rev.append(data_rev_column)

        # Convert data_rev to an array and return the transpose since it contains the reverted columns of data as rows
        reverted_data = np.array(data_rev)
        if converted_shape:
            # Convert back to (n,)
            return reverted_data.flatten()
        return reverted_data.T


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
