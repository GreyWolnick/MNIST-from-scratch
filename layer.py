import utlities as util

class LearnableParameter:
    """
    Represents a learnable parameter in a neural network, such as weights or biases.

    Attributes:
        param (vector/matrix): The parameter's values.
        shape (tuple): The dimensions of the parameter.
        grad (vector/matrix or None): The gradient of the parameter; initialized to None.
        m (vector/matrix): The first moment estimate for Adam optimizer; initialized to zeros.
        v (vector/matrix): The second moment estimate for Adam optimizer; initialized to zeros.
    """

    def __init__(self, init_param, shape):
        self.param = init_param
        self.shape = shape
        self.grad = None
        self.m = util.create_zeros(shape)
        self.v = util.create_zeros(shape)

class Layer:
    """
    Base class for all layers in a neural network.

    Attributes:
        network (Network or None): The network to which this layer belongs; initialized to None.
    """

    def __init__(self):
        self.network = None

    def parameters(self) -> list[LearnableParameter]:
        """
        Returns the learnable parameters of the layer.

        Returns an empty list by default, as some layers may have no learnable parameters.
        """
        return []

    def forward(self, inputs):
        """
         Defines the forward pass computation for the layer.

         This method should be implemented by subclasses to define the specific computation.
         """
        raise NotImplementedError

    def backward(self, forward_inputs, backward_inputs):
        """
         Defines the backward pass computation for the layer.

         This method should be implemented by subclasses to define the specific computation.
         """
        raise NotImplementedError

    def __call__(self, inputs: list[list[float]]) -> list[list[float]]:
        """
        Computes the forward pass for the layer and records the operation in the network's operation history.
        """
        outputs = self.forward(inputs)
        self.network.record_operation(inputs, self.backward)
        return outputs

class Linear(Layer):
    """
    Implementation of a fully-connected linear layer.

    This layer performs the operation Y = XW + b, where:
        - X is the input matrix, shape: (batch_size, input_features)
        - W is the weight matrix initialized with Xavier initialization, shape: (input_features, output_features)
        - b is the bias vector initialized with Xavier initialization, shape: (1, output_features)
        - Y is the output matrix, shape: (batch_size, output_features)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._W = LearnableParameter(util.init_weight_matrix(in_features, out_features), (in_features,out_features))
        self._b = LearnableParameter(util.init_bias_vector(in_features, out_features), (out_features,))

    def parameters(self) -> list[LearnableParameter]:
        """
        Return the learnable parameters of the layer.
        """
        return [self._W, self._b]

    def forward(self, batch: list[list[float]]) -> list[list[float]]:
        """
        Perform the forward pass by computing the linear transformation Y = XW + b.

        Parameters:
        batch: The input data for the layer, where each row represents the input features of a data sample.
        """
        W = self._W.param
        b = self._b.param

        batch_size, input_features = len(batch), len(batch[0])
        output_features = len(W[0])

        Y = [
            [
                sum(batch[i][k] * W[k][j] for k in range(input_features)) + b[j]
                for j in range(output_features)
            ]
            for i in range(batch_size)
        ]

        return Y

    def backward(self, forward_inputs: list[list[float]], backward_inputs: list[list[float]]) -> list[list[float]]:
        """
        Compute the gradients of the loss with respect to each learnable parameter.
        Compute the gradient of the loss with respect to the inputs and propagate backwards.

        Parameters:
        forward_inputs:  The inputs that were passed through the linear layer during forward propagation.
        backward_inputs: The gradients of the loss with respect to the outputs of this layer.

        Updates:
        - The gradient of the learnable weight matrix (_W.grad)
        - The gradient of the learnable bias vector (_b.grad)

        Returns:
        The gradient of the loss with respect to the inputs of this layer.
        """
        self._W.grad = util.matrix_multiply(util.transpose(forward_inputs), backward_inputs)
        self._b.grad = [sum(backward_inputs[i][j] for i in range(len(backward_inputs))) for j in range(len(backward_inputs[0]))]

        return util.matrix_multiply(backward_inputs, util.transpose(self._W.param))


class ReLU(Layer):
    """
    Implementation of a ReLU layer.

    The ReLU activation function is defined as:
        f(x) = max(0, x)
    where x is a scalar.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch: list[list[float]]) -> list[list[float]]:
        """
        Apply the ReLU activation function to each feature across a batch.

        Parameters:
        batch: The input data for the layer, where each row represents the input features of a data sample.
        """
        input_features = len(batch[0])
        return [[max(0.,batch[i][j]) for j in range(input_features)] for i in range(len(batch))]

    def backward(self, forward_inputs: list[list[float]], backward_inputs: list[list[float]]) -> list[list[float]]:
        """
        Compute the gradient of the loss function with respect to the inputs during backpropagation.

        Parameters:
        forward_inputs: The inputs that were passed through the ReLU layer during forward propagation.
        backward_inputs: The gradient of the loss with respect to the outputs of this layer.
        """
        input_features = len(forward_inputs[0])
        return [
            [backward_inputs[i][j] if forward_inputs[i][j] > 0 else 0. for j in range(input_features)]
            for i in range(len(forward_inputs))
        ]

