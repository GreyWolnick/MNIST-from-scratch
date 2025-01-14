from layer import Layer, LearnableParameter

class NeuralNetwork:
    """
    The base class representing a neural network, managing layers, parameters, and backpropagation.

    Attributes:
        _parameters: Store all learnable parameters of the network for optimization.
        _grad_stack: A stack to record operations for use during backpropagation.
    """

    def __init__(self):
        self._parameters = []
        self._grad_stack = []

    def __setattr__(self, name: str, value: any):
        """
        Automatically sets the network attribute for layers.

        If a new attribute is set that is an instance of Layer:
        - assign the current network to the layer
        - extend the parameter list to include the layer's learnable parameters
        """
        super().__setattr__(name, value)
        if isinstance(value, Layer):
            value.network = self
            self._parameters.extend(value.parameters())

    def parameters(self) -> list[LearnableParameter]:
        """
        Returns the learnable parameters of the network.
        """
        return self._parameters

    def record_operation(self, inputs: list[list[float]], grad_fn: callable):
        """
        Records forward pass operations on the _grad_stack to be traversed during backpropagation.
        """
        self._grad_stack.append((inputs, grad_fn))

    def forward(self, inputs):
        """
        Defines the forward pass computation through the layers of the neural network.

        This method should be implemented by subclasses to define the specific computation.
        """
        raise NotImplementedError

    def backward(self, backward_inputs: list[list[float]]):
        """
        Performs backpropagation through the network by iterating through the recorded operations in reverse order,
            passing necessary inputs and gradients into the corresponding gradient functions.
        """
        while len(self._grad_stack):
            forward_inputs, grad_fn = self._grad_stack.pop()
            backward_inputs = grad_fn(forward_inputs, backward_inputs)

