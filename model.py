import tensorflow as tf
import tensorflow.contrib.layers as layer
from data import ToyData
from args import get_args

arg = get_args()

batch = arg.batch_size
z_dim = arg.z_dim
x_dim = arg.x_dim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def dense(input, out_num, activation_fn=None, scope='dense', batch_norm=False, is_tr=None):
    scope1 = scope
    scope2 = scope + '_bn'
    scope3 = scope + '_relu'
    fc = layer.fully_connected(input, out_num, activation_fn=None, scope=scope1)
    if batch_norm:
        bn = layer.batch_norm(fc, activation_fn=activation_fn, is_training=is_tr, scope=scope2)

        return bn

    if activation_fn is not None:
        out = activation_fn(fc, name=scope3)
        return out

    return fc


# input: real image x -> return: latent space z
# for toy example : input: (N, 2) -> return: latent z (N,2)
# encoder
# fully -> dependent layer
# split layer
# encoder
def Q(x, is_tr=None):
    with tf.variable_scope('GQ'):
        fc1 = dense(x, arg.E_layer, activation_fn=tf.nn.relu, scope='fc1', batch_norm=True,
                    is_tr=is_tr)
        fc2 = dense(fc1, arg.E_layer, activation_fn=tf.nn.relu, scope='fc2', batch_norm=True,
                    is_tr=is_tr)
        if arg.param_trick == 'split':
            fc3 = dense(fc2, 2 * z_dim, activation_fn=tf.nn.relu, scope='fc3')
            mu, log_sig_sq = tf.split(fc3, 2, axis=1)
        else:
            mu = dense(fc2, z_dim, scope='mu')
            log_sig_sq = dense(fc2, z_dim, scope='log_sig_sq')
        e = tf.random_normal([batch, z_dim])
        out = tf.add(mu, tf.multiply(tf.exp(log_sig_sq / 2), e))
    return out


# input: latent z -> return: fake image x
# for toy example : input: z-(N, 2) -> return: fake x (N,2)
# decoder
def P(z, is_tr):
    with tf.variable_scope('GP'):
        fc1 = dense(z, arg.G_layer, activation_fn=tf.nn.relu, scope='fc1', batch_norm=True,
                    is_tr=is_tr)
        fc2 = dense(fc1, arg.G_layer, activation_fn=tf.nn.relu, scope='fc2', batch_norm=True,
                    is_tr=is_tr)
        fc3 = dense(fc2, arg.G_layer, activation_fn=tf.nn.relu, scope='fc3', batch_norm=True,
                    is_tr=is_tr)
        out = dense(fc3, x_dim, activation_fn=None, scope='out')
    return out


# input -> x, z ==> concat
def D(x, z, is_tr):
    with tf.variable_scope('D'):
        input = tf.concat([x, z], axis=1)
        fc1 = dense(input, arg.D_layer, activation_fn=tf.nn.relu, scope='fc1', batch_norm=True,
                    is_tr=is_tr)
        fc2 = dense(fc1, arg.D_layer, activation_fn=tf.nn.relu, scope='fc2', batch_norm=True,
                    is_tr=is_tr)
        fc3 = dense(fc2, arg.D_layer, activation_fn=tf.nn.relu, scope='fc3', batch_norm=True,
                    is_tr=is_tr)
        out = dense(fc3, 1, activation_fn=tf.nn.sigmoid, scope='out')
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
