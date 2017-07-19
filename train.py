from model import Q, P, D
from data import ToyData
import tensorflow as tf
import numpy as np

# lr = 1e-3
batch = 30
g_lr = 0.000004
d_lr = 0.000002
z_dim = 2
x_dim= 2


class ALI:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 2])
        self.z = tf.placeholder(tf.float32, [None, z_dim])

        self.z_hat = Q(self.x)
        self.x_hat = P(self.z)

        with tf.variable_scope("Discriminator") as scope:
            self.D_enc = D(self.x, self.z_hat)  # D(x,Gz(x))
            scope.reuse_variables()
            self.D_gen = D(self.x_hat, self.z)  # D(Gx(z), z)

        # self.D_loss = -tf.reduce_mean(self.log(self.D_enc) + self.log(1 - self.D_gen))
        # self.G_loss = -tf.reduce_mean(self.log(self.D_gen) + self.log(1 - self.D_enc))

        # self.D_loss = -tf.reduce_mean(self.D_enc**2 + (1 - self.D_gen)**2)
        # self.G_loss = -tf.reduce_mean(self.D_gen**2 + (1 - self.D_enc)**2)

        # ls - wrong
        # self.D_loss = 0.5 * tf.reduce_mean(self.D_enc ** 2 + (1 - self.D_gen) ** 2)
        # self.G_loss = 0.5 * tf.reduce_mean(self.D_gen ** 2 + (1 - self.D_enc) ** 2)

        # ls -right way
        self.D_loss = 0.5 * tf.reduce_mean((1 + self.D_gen) ** 2 + (1 - self.D_enc) ** 2)
        self.G_loss = 0.5 * tf.reduce_mean((1 + self.D_enc) ** 2 + (1 - self.D_gen) ** 2)


        # self.theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        # self.theta_G_P = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='P')
        # self.theta_G_Q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q')

        self.theta_D = [v for v in tf.all_variables() if v.name.startswith('D')]

        self.theta_G = [v for v in tf.all_variables() if v.name.startswith('G')]

        self.D_train = tf.train.AdamOptimizer(learning_rate=d_lr).minimize(self.D_loss,
                                                                           var_list=self.theta_D)
        self.G_train = tf.train.AdamOptimizer(learning_rate=g_lr).minimize(self.G_loss,
                                                                           var_list=self.theta_G)

        self.data = ToyData(load_saved=False, batch_size=batch)

        self.save = True
        self.load = False

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()

    @staticmethod
    def log(x):
        return tf.log(x + 1e-8)

    @staticmethod
    def sample_z(m, n):
        return np.random.normal(size=[m, n])

    def train(self):

        self.sess.run(self.init)
        saver = tf.train.Saver()

        if self.load:
            saver = tf.train.import_meta_graph('ali_model.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        for i in range(500000):
            x_feed = self.data.get_next_batch()
            z_feed = self.sample_z(len(x_feed), z_dim)

            loss_d, train_d = self.sess.run([self.D_loss, self.D_train],
                                            feed_dict={self.x: x_feed, self.z: z_feed})
            loss_g, train_g = self.sess.run([self.G_loss, self.G_train],
                                            feed_dict={self.x: x_feed, self.z: z_feed})

            if i % 100 == 0:
                print("iter : ", i, " D loss : ", loss_d, " G loss : ", loss_g)

            if i % 10000 == 0:
                name = './figs/' + str(i) + 'th_ls'
                self.test(savefig=True,figname=name)

        saver.save(self.sess, 'ali_model')

    def test(self,savefig=False,figname=None):

        if self.load:
            saver = tf.train.import_meta_graph('ali_model.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        draw_list_x = []
        draw_list_z = []
        for i in range(80):
            feed_x = self.data.get_next_batch()
            z = self.sess.run(self.z_hat, feed_dict={self.x: feed_x})
            x = self.sess.run(self.x_hat, feed_dict={self.z: z})
            draw_list_x.append(x)
            draw_list_z.append(z)

        draw_list_x = np.asarray(draw_list_x)
        draw_list_z = np.asarray(draw_list_z)
        draw_list_x = draw_list_x.reshape([-1, 2])
        draw_list_z = draw_list_z.reshape([-1, 2])

        if savefig:
            x_name = figname + '_x.png'
            z_name = figname + '_z.png'
            self.data.draw_x(draw_list_x, save=True, figname=x_name)
            self.data.draw_x(draw_list_z, save=True, figname=z_name)
        else:
            self.data.draw_x(draw_list_z)

        # z_l1 = self.sess.run(self.z_hat, feed_dict={self.x: l1})
        # x_l1 = self.sess.run(self.x_hat, feed_dict={self.z: z_l1})
        #
        # z_l5 = self.sess.run(self.z_hat, feed_dict={self.x: l5})
        # x_l5 = self.sess.run(self.x_hat, feed_dict={self.z: z_l5})
        #
        # z_l6 = self.sess.run(self.z_hat, feed_dict={self.x: l6})
        # x_l6 = self.sess.run(self.x_hat, feed_dict={self.z: z_l6})


if __name__ == "__main__":
    model = ALI()

    model.train()
    model.test()
