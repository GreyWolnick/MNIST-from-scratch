import math
import utlities as util
from neural import NeuralNetwork
from layer import LearnableParameter
from utlities import scalar_matrix_multiply, matrix_add, matrix_divide, matrix_subtract

class CrossEntropyLoss:
    """
    Implementation of the combined softmax and cross-entropy loss function for a batch.

    Attributes:
        network: The network to which the cross entropy loss operations should be recorded.
    """

    def __init__(self, network: NeuralNetwork):
        self.network = network

    def __call__(self, inputs: list[list[float]], targets: list[int]):
        """
        Compute the combined softmax and cross-entropy loss across a batch.

        Parameters:
        - inputs: The pre-softmax outputs (logits) from the network.
        - targets: The ground truth labels for each sample in the batch.

        Returns the average cross-entropy loss across the batch.
        """
        # Precalculate softmax to save computation in the backwards pass
        softmaxed = [self.softmax(logits) for logits in inputs]
        self.network.record_operation(softmaxed, self.backward)

        # Compute the CE loss
        loss = sum([-math.log(softmaxed[i][target]) for i, target in enumerate(targets)])
        loss = loss / len(inputs) # Average CE loss across the batch

        return loss

    def softmax(self, logits: list[float]) -> list[float]:
        """
        Compute the softmax of each logit for a given sample.
        """
        max_logit = max(logits)  # Stabilize logits to prevent overflow
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        sum_exp_logits = sum(exp_logits)

        return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

    def backward(self, forward_inputs: list[list[float]], targets: list[int]) -> list[list[float]]:
        """
        Compute the gradient of the loss with respect to the pre-softmax inputs.

        Parameters:
        - forward_inputs: The softmaxed outputs from the forward pass (pre-softmaxed during the forward pass for efficiency).
        - targets: The ground truth label for each sample in the batch.

        Returns:
        The gradients of the loss with respect to the inputs.
        """
        grad = forward_inputs.copy()
        for i, target in enumerate(targets):
            grad[i][target] -= 1

        return grad

class Adam:
    """
    Implementation of the Adam optimizer for a learnable parameter.

    Attributes:
        params: A list of learnable parameters for a given neural network.
        lr: The learning rate for the optimizer (default: 0.001).
        betas: The beta coefficients for the first and second moment estimates (default: (0.9, 0.999)).
        eps: A small constant to prevent division by zero during optimization (default: 1e-08).
    """

    def __init__(self, params: list[LearnableParameter], lr=0.001, betas=(0.9,0.999), eps=1e-08):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 1

    def step(self):
        """
        Perform an optimization step for each learnable parameter.
        Separately handles vector vs matrix parameters
        """
        for param in self.params:
            if len(param.shape) == 1:
                self.vector_step(param)
            elif len(param.shape) == 2:
                self.matrix_step(param)
        self.t += 1

    def vector_step(self, param: LearnableParameter):
        """
        Performs a single Adam step for a vector learnable parameter.
        """
        # Compute first moment estimate
        param.m = [(self.b1 * param.m[i]) + ((1-self.b1) * param.grad[i]) for i in range(len(param.m))]

        # Compute second moment estimate
        param.v = [(self.b2 * param.v[i]) + ((1-self.b2) * param.grad[i]**2) for i in range(len(param.v))]

        # Compute bias corrected estimates
        m_hat = [m_i / (1-self.b1**self.t) for m_i in param.m]
        v_hat = [v_i / (1 - self.b2 ** self.t) for v_i in param.v]

        # Update parameter
        for i in range(len(param.param)):
            param.param[i] -= self.lr * (m_hat[i] / (v_hat[i]**0.5 + self.eps))

    def matrix_step(self, param: LearnableParameter):
        """
        Performs a single Adam step for a matrix learnable parameter.
        """
        # Compute first moment estimate
        beta_m = scalar_matrix_multiply(self.b1, param.m)
        inv_beta_grad = scalar_matrix_multiply(1 - self.b1, param.grad)
        param.m = matrix_add(beta_m, inv_beta_grad)

        # Compute second moment estimate
        beta_v = scalar_matrix_multiply(self.b2, param.v)
        inv_beta_grad_sqrd = scalar_matrix_multiply(1 - self.b2, util.scalar_matrix_exp(2, param.grad))
        param.v = matrix_add(beta_v, inv_beta_grad_sqrd)

        # Compute bias corrected estimates
        m_hat = util.scalar_divides_matrix(1 - (self.b1 ** self.t), param.m)
        v_hat = util.scalar_divides_matrix(1 - (self.b2 ** self.t), param.v)

        # Update parameter
        moment_estimate = util.matrix_divide(m_hat, util.scalar_matrix_exp(0.5, v_hat), stability_term=self.eps)
        param.param = util.matrix_subtract(param.param, util.scalar_matrix_multiply(self.lr, moment_estimate))
