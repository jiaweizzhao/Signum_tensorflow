# Signum_tensorflow
## This is the repository for Signum optimizer implemented in TensorFlow.
### see the detailed discription of Signum in the original paper at: https://arxiv.org/abs/1711.05101

Arguments:\
        lr (float): learning rate\
        momentum (float, optional): momentum factor (default: 0.9)\
        weight_decay (float, optional): weight decay (default: 0)

    Example:
        >>> import tensorflow as tf
        >>> your_loss = (eg: tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv))
        >>> optimizer = signum.Signum(lr=0.01, momentum=0.9, weight_decay = 0).minimize(your_loss)
        >>> sess = tf.Session()
        >>> loss, _ = sess.run([your_loss, optimizer])

Note:\
        The optimizer updates the weight by:\
            momentum = beta * momentum + (1-beta)*rescaled_grad\
            weight = (1 - lr * weight_decay) * weight - lr * sign(momentum)

Considering the specific case of Momentum, the update Signum can be written as

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20g_t%20%26%3D%20%5Cnabla%20J%28W_%7Bt-1%7D%29%5C%5C%20m_t%20%26%3D%20%5Cbeta%20m_%7Bt-1%7D%20&plus;%20%281%20-%20%5Cbeta%29%20g_t%5C%5C%20W_t%20%26%3D%20W_%7Bt-1%7D%20-%20%5Ceta_t%20%5Ctext%7Bsign%7D%28m_t%29%20%5Cend%7Balign*%7D)

If do not consider Momentum, the update Sigsgd can be written as

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20g_t%20%26%3D%20%5Cnabla%20J%28W_%7Bt-1%7D%29%5C%5C%20W_t%20%26%3D%20W_%7Bt-1%7D%20-%20%5Ceta_t%20%5Ctext%7Bsign%7D%28g_t%29%20%5Cend%7Balign*%7D)

Description of example:\
Tested on a standard TensorFlow example of MNIST dataset. (https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network.py).\
train.py: train and validate the dataset\
signum.py: contain the signum optimizer
