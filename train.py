from model import Q, P, D
from data import ToyData
import tensorflow as tf
import numpy as np
from args import get_args
import os
from time import localtime, strftime

# lr = 1e-3
arg = get_args()


class ALI:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 2])
        self.z = tf.placeholder(tf.float32, [None, arg.z_dim])

        if arg.batch_norm:
            self.is_tr = tf.placeholder(tf.bool)
        else:
            self.is_tr = None

        self.z_hat = Q(self.x, is_tr=self.is_tr)
        self.x_hat = P(self.z, is_tr=self.is_tr)

        with tf.variable_scope("Discriminator") as scope:
            self.D_enc = D(self.x, self.z_hat, is_tr=self.is_tr)  # D(x,Gz(x))
            scope.reuse_variables()
            self.D_gen = D(self.x_hat, self.z, is_tr=self.is_tr)  # D(Gx(z), z)

        # self.D_loss = -tf.reduce_mean(self.log(self.D_enc) + self.log(1 - self.D_gen))
        # self.G_loss = -tf.reduce_mean(self.log(self.D_gen) + self.log(1 - self.D_enc))

        # self.D_loss = -tf.reduce_mean(self.D_enc**2 + (1 - self.D_gen)**2)
        # self.G_loss = -tf.reduce_mean(self.D_gen**2 + (1 - self.D_enc)**2)

        # ls - wrong
        # self.D_loss = 0.5 * tf.reduce_mean(self.D_enc ** 2 + (1 - self.D_gen) ** 2)
        # self.G_loss = 0.5 * tf.reduce_mean(self.D_gen ** 2 + (1 - self.D_enc) ** 2)

        #
        # self.D_loss = 0.5 * tf.reduce_mean((1 + self.D_gen) ** 2 + (1 - self.D_enc) ** 2)
        # self.G_loss = 0.5 * tf.reduce_mean((1 + self.D_enc) ** 2 + (1 - self.D_gen) ** 2)

        # ls -right way
        if arg.loss == 'ls':
            self.D_loss = 0.5 * tf.reduce_mean(self.D_enc ** 2 + (1 - self.D_gen) ** 2)
            self.G_loss = 0.5 * tf.reduce_mean(self.D_gen ** 2 + (1 - self.D_enc) ** 2)

        else:
            self.D_loss = -tf.reduce_mean(self.log(self.D_enc) + self.log(1 - self.D_gen))
            self.G_loss = -tf.reduce_mean(self.log(self.D_gen) + self.log(1 - self.D_enc))

        self.theta_D = [v for v in tf.all_variables() if v.name.startswith('D')]

        self.theta_G = [v for v in tf.all_variables() if v.name.startswith('G')]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_train = tf.train.AdamOptimizer(learning_rate=arg.d_lr).minimize(self.D_loss,
                                                                                   var_list=self.theta_D)
            self.G_train = tf.train.AdamOptimizer(learning_rate=arg.g_lr).minimize(self.G_loss,
                                                                                   var_list=self.theta_G)

        self.data = ToyData(load_saved=False, batch_size=arg.batch_size)

        self.save = True
        # self.load = False

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()

        result_dir = os.path.join(os.getcwd(), 'result')
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        self.path = os.path.join(os.getcwd(), 'result', strftime("%m-%d-%H-%M", localtime()))
        os.mkdir(self.path)
        os.mkdir(self.path + '/figs')

        f = open(os.path.join(self.path, 'info.txt'), 'w')
        tuples = vars(arg).items()
        for x in tuples:
            f.write(str(x))
            f.write('\n')
        f.close()

    @staticmethod
    def log(x):
        return tf.log(x + 1e-8)

    @staticmethod
    def sample_z(m, n):
        return np.random.normal(size=[m, n])

    def train(self):

        self.sess.run(self.init)
        saver = tf.train.Saver()

        # if self.load:
        #     saver = tf.train.import_meta_graph('ali_model.meta')
        #     saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        for i in range(arg.iter):
            x_feed = self.data.get_next_batch()
            z_feed = self.sample_z(len(x_feed), arg.z_dim)

            if arg.batch_norm:
                loss_d, train_d = self.sess.run([self.D_loss, self.D_train],
                                                feed_dict={self.x: x_feed, self.z: z_feed,
                                                           self.is_tr: True})
                loss_g, train_g = self.sess.run([self.G_loss, self.G_train],
                                                feed_dict={self.x: x_feed, self.z: z_feed,
                                                           self.is_tr: True})
            else:
                loss_d, train_d = self.sess.run([self.D_loss, self.D_train],
                                                feed_dict={self.x: x_feed, self.z: z_feed})
                loss_g, train_g = self.sess.run([self.G_loss, self.G_train],
                                                feed_dict={self.x: x_feed, self.z: z_feed})

            if i % 100 == 0:
                print("iter : ", i, " D loss : ", loss_d, " G loss : ", loss_g)

            if i % 10000 == 0:
                name = self.path + '/figs/' + str(i) + 'th_ls'
                self.test(savefig=True, figname=name)

        saver.save(self.sess, self.path + '/ali_model')

    def test(self, savefig=False, figname=None):

        # if self.load:
        #     saver = tf.train.import_meta_graph('ali_model.meta')
        #     saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        total_x = []
        total_z = []

        draw_list_x = []
        draw_list_z = []

        test_data = self.data.get_labeled_data()
        for i in test_data:
            draw_list_x.clear()
            draw_list_z.clear()
            start = 0
            end = arg.batch_size
            while end <= len(i):
                if arg.batch_norm:
                    z = self.sess.run(self.z_hat, feed_dict={self.x: i[start:end],
                                                             self.is_tr: False})
                    x = self.sess.run(self.x_hat, feed_dict={self.z: z, self.is_tr: False})

                else:
                    z = self.sess.run(self.z_hat, feed_dict={self.x: i[start:end]})
                    x = self.sess.run(self.x_hat, feed_dict={self.z: z})

                start += arg.batch_size
                end += arg.batch_size
                draw_list_x.extend(x)
                draw_list_z.extend(z)

            draw_list_x1 = np.asarray(draw_list_x)
            draw_list_z1 = np.asarray(draw_list_z)

            total_x.append(draw_list_x1)
            total_z.append(draw_list_z1)

        total_x = np.asarray(total_x)
        total_z = np.asarray(total_z)

        if savefig:
            x_name = figname + '_x.png'
            z_name = figname + '_z.png'
            self.data.draw_x(total_x, save=True, figname=x_name)
            self.data.draw_x(total_z, save=True, figname=z_name)
        else:
            self.data.draw_x(draw_list_z)


if __name__ == "__main__":
    model = ALI()

    model.train()
    model.test()
