import tensorflow as tf
import tensorflow.contrib.layers as layer
from data import ToyData
from args import get_args

arg = get_args()
batch = arg.batch_size
z_dim = 2
input_dim = 2


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# input: real image x -> return: latent space z
# for toy example : input: (N, 2) -> return: latent z (N,2)
# encoder
# fully -> dependent layer
'''
def Q(x):
    with tf.variable_scope('GQ'):
        fc1 = layer.fully_connected(x, 400, activation_fn=tf.nn.relu, scope='fc1')
        fc2 = layer.fully_connected(fc1, 400, activation_fn=tf.nn.relu, scope='fc2')
        mu = layer.fully_connected(fc2, z_dim, activation_fn=None, scope='mu')
        log_sig_sq = layer.fully_connected(fc2, z_dim, activation_fn=None, scope='log_sig_sq')
        e = tf.random_normal([batch, z_dim])
        out = tf.add(mu, tf.multiply(tf.exp(log_sig_sq / 2), e))
    return out
'''


# split layer
def Q(x):
    with tf.variable_scope('GQ'):
        fc1 = layer.fully_connected(x, 400, activation_fn=tf.nn.relu, scope='fc1')
        fc2 = layer.fully_connected(fc1, 400, activation_fn=tf.nn.relu, scope='fc2')
        fc3 = layer.fully_connected(fc2, 400, activation_fn=tf.nn.relu, scope='fc2')
        # fc3 = layer.fully_connected(fc2, 2 * z_dim, activation_fn=tf.nn.relu, scope='fc3')
        # mu, log_sig_sq = tf.split(fc3, 2, axis=1)
        mu = layer.fully_connected(fc3, z_dim, activation_fn=None, scope='mu')
        log_sig_sq = layer.fully_connected(fc3, z_dim, activation_fn=None, scope='log_sig_sq')
        e = tf.random_normal([batch, z_dim])
        out = tf.add(mu, tf.multiply(tf.exp(log_sig_sq / 2), e))
    return out


# input: latent z -> return: fake image x
# for toy example : input: z-(N, 2) -> return: fake x (N,2)
# decoder
def P(z):
    with tf.variable_scope('GP'):
        fc1 = layer.fully_connected(z, 400, activation_fn=tf.nn.relu, scope='fc1')
        fc2 = layer.fully_connected(fc1, 400, activation_fn=tf.nn.relu, scope='fc2')
        fc3 = layer.fully_connected(fc2, 400, activation_fn=tf.nn.relu, scope='fc3')
        fc4 = layer.fully_connected(fc3, 400, activation_fn=tf.nn.relu, scope='fc3')
        out = layer.fully_connected(fc4, input_dim, activation_fn=None, scope='out')
    return out


# input -> x, z ==> concat
def D(x, z):
    with tf.variable_scope('D'):
        input = tf.concat([x, z], axis=1)
        fc1 = layer.fully_connected(input, 200, activation_fn=tf.nn.relu, scope='fc1')
        fc2 = layer.fully_connected(fc1, 200, activation_fn=tf.nn.relu, scope='fc2')
        fc3 = layer.fully_connected(fc2, 200, activation_fn=tf.nn.relu, scope='fc3')
        fc4 = layer.fully_connected(fc3, 200, activation_fn=tf.nn.relu, scope='fc3')
        out = layer.fully_connected(fc4, 1, activation_fn=tf.nn.sigmoid, scope='out')
    return out


'''
shape test area
'''

# '''
if __name__ == "__main__":
    a = ToyData(load_saved=True)
    x_feed = a.get_next_batch()

    input_x = tf.placeholder(tf.float32, [None, 2], name='input')

    # out = Q(input_x)
    out = Q(input_x)

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    o = sess.run(out, feed_dict={input_x: x_feed})
    # print(m.shape)
    # print(l.shape)
    # print(e.shape)
    print(o.shape)
# '''
