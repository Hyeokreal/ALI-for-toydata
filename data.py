import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_args


class ToyData:
    def __init__(self, load_saved=False, path='./data', batch_size=32):
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        self.path = path
        self.path_d = '/data_raw.npy'
        self.path_c = '/data_raw_class.npy'
        self.args = get_args()
        self.data, self.data_c = self.load_data(load_saved, path)
        self.data_list = list(self.data)


    def load_data(self, load_saved, path):
        if load_saved:
            data, data_c = np.load(path + self.path_d), np.load(path + self.path_c)
            return data, data_c
        else:
            return self.generate_data(number=self.args.num_dots)

    def generate_data(self,mixtures=4, number=100):
        means = list()

        sq = np.sqrt(mixtures)

        for x in range(-int(sq) + 1, int(sq), 2):
            for y in range(-int(sq) + 1, int(sq), 2):
                means.append([x, y])
                # print(x, y)

        # from bottom left
        # y up direction ordered
        classes = list()
        for i in range(mixtures):
            classes.append(np.random.multivariate_normal(means[i],
                                                         [[0.001, 0],
                                                          [0, 0.001]], number))
        x_1 = np.asarray(classes)
        x = x_1.reshape((-1, 2))

        print(x_1.shape)
        print(x.shape)

        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        np.save(self.path+self.path_d, x)
        np.save(self.path+self.path_c, x_1)
        return x, x_1

    def draw_x(self, given_data=None, save=False, figname=None):

        f, ax = plt.subplots(1)

        if given_data is not None:
            data = given_data

        else:
            data = self.data

        cmap = plt.cm.get_cmap('summer', data.shape[0])
        for ix in range(data.shape[0]):
            xy = data[ix].T
            x_pos = xy[0]
            y_pos = xy[1]
            ax.scatter(x_pos, y_pos, c=cmap(ix), marker='.')

        if save:
            f.savefig(figname)
        else:
            plt.show()
        plt.close()

    def get_next_batch(self):

        r = self.data[self.start:self.end]

        if self.end + self.batch_size > len(self.data) - 1:
            self.start = 0
            self.end = self.batch_size

        else:
            self.start += self.batch_size
            self.end += self.batch_size

        return r

    def get_labeled_data(self):
        return self.data_c


if __name__ == '__main__':
    data = ToyData(load_saved=False, path='./data')
    data.draw_x(given_data=data.data_c, save=True,figname='./fuck.png')