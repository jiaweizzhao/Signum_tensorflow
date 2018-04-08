from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class Signum(optimizer.Optimizer):
    r"""Implements Signum optimizer that takes the sign of gradient or momentum.

        See details in the original paper at:https://arxiv.org/abs/1711.05101

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0.9)
            weight_decay (float, optional): weight decay (default: 0)

        Example:
            >>> import tensorflow as tf
            >>> your_loss = (eg: tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv))
            >>> optimizer = signum.Signum(lr=0.01, momentum=0.9, weight_decay = 0).minimize(your_loss)
            >>> sess = tf.Session()
            >>> loss, _ = sess.run([your_loss, optimizer])

        .. note::
            The optimizer updates the weight by:
                buf = momentum * buf + (1-momentum)*rescaled_grad
                weight = (1 - lr * weight_decay) * weight - lr * sign(buf)

            Considering the specific case of Momentum, the update Signum can be written as

            .. math::
                    \begin{split}g_t = \nabla J(W_{t-1})\\
    			    m_t = \beta m_{t-1} + (1 - \beta) g_t\\
    				W_t = W_{t-1} - \eta_t \text{sign}(m_t)}\end{split}

            where p, g, v and :math:`\rho` denote the parameters, gradient,
            velocity, and momentum respectively.

            If do not consider Momentum, the update Sigsgd can be written as

            .. math::
                	g_t = \nabla J(W_{t-1})\\
    				W_t = W_{t-1} - \eta_t \text{sign}(g_t)}

        """

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0, use_locking=False, name="Signum"):
        super(Signum, self).__init__(use_locking, name)
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay

        if momentum == 0:
            self.momentum_use = False
        else:
            self.momentum_use = True

        if weight_decay == 0:
            self.weight_decay_use = False
        else:
            self.weight_decay_use = True

        # Tensorazing optimizer parameters 
        self._lr_t = None
        self._momentum_t = None
        self._weight_decay_t = None

    def _prepare(self): #Create the optimizer parameters 
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._momentum_t = ops.convert_to_tensor(self._momentum, name="momentum")
        self._weight_decay_t = ops.convert_to_tensor(self._weight_decay, name="weight_decay")

    def _create_slots(self, var_list): #Create and initialise momentum, an optimizer variable.
        if self.momentum_use:
            for v in var_list:
                self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var): #Update the training variables and momentum
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)
        weight_decay_t = math_ops.cast(self._weight_decay_t, var.dtype.base_dtype)
        
        #signum
        if self.momentum_use: #Use the Signum to caculate variables and momentum
            m = self.get_slot(var, "m")
            m_t = m.assign(math_ops.mul(momentum_t, m) + math_ops.mul((1-momentum_t), grad))
            if self.weight_decay_use:
                decay = 1 - math_ops.mul(lr_t, weight_decay_t)
                var = var.assign(var * decay)
            var_update = state_ops.assign_sub(var, math_ops.mul(lr_t, tf.sign(m_t)))
            return control_flow_ops.group(*[var_update, m_t])

        #signsgd
        else:#Use Signsgd to caculate variables
            var_update = state_ops.assign_sub(var, math_ops.mul(lr_t, tf.sign(grad)))
            return control_flow_ops.group(*[var_update])
