import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ToyData:
    def __init__(self, load_saved=False, path_d='./toydata4.npy',
                 path_c='./toydata_class4.npy', batch_size=32):
        self.data, self.data_c = self.load_data(load_saved, path_d, path_c)
        self.data_list = list(self.data)
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size

    def load_data(self, load_saved, path_d=None, path_c=None):
        if load_saved:
            data, data_c = np.load(path_d), np.load(path_c)
            return data, data_c
        else:
            return self.generate_data()

    @staticmethod
    def generate_data(mixtures=4, number=100):
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
        # print(x.shape)
        np.save('./toydata4.npy', x)
        np.save('./toydata_class4.npy', x_1)
        return x, x_1

    def draw_x(self, draw_data=None,save=False,figname=None):

        f, ax = plt.subplots(1)

        if draw_data is not None:
            data = draw_data

        else:
            data = self.data

        cmap = plt.cm.get_cmap('summer', data.shape[0])
        for ix in range(data.shape[0]):
            xy = data[ix].T
            # print(xy.shape)
            x_pos = xy[0]
            y_pos = xy[1]
            ax.scatter(x_pos, y_pos, c=cmap(ix), marker='.')

        if save:
            plt.savefig(figname)
        else:
            plt.show()
        plt.close()

    def get_next_batch(self):

        r = self.data[self.start:self.end]
        # print("start: ", self.start, " end: ", self.end)

        # if self.end == len(self.data)-1:
        if self.end + self.batch_size > len(self.data) - 1:
            self.start = 0
            self.end = self.batch_size

        else:
            self.start += self.batch_size
            self.end += self.batch_size

        return r

    def get_labeled_data(self, label, number):

        start = label * 100
        end = start + number

        return self.data[start:end]


if __name__ == '__main__':
    data = ToyData(load_saved=False, path_d='./toydata4.npy', path_c='./todydata_class4.npy')
    # label_0 = data.get_labeled_data(0, 32)
    # label_1 = data.get_labeled_data(1, 32)
    # label_2 = data.get_labeled_data(2, 32)
    # label_3 = data.get_labeled_data(3, 32)
    # label_4 = data.get_labeled_data(4, 32)
    # label_5 = data.get_labeled_data(5, 32)
    # label_6 = data.get_labeled_data(6, 32)
    #
    # a = []
    # a.append(label_0)
    # a.append(label_1)
    # a.append(label_5)
    # a.append(label_6)
    # a = np.asarray(a)
    # print(a.shape)
    # a = a.reshape([-1,2])
    # print(a.shape)
    data.draw_x()



    # for i in range(2500):
    #     print(data.get_next_batch().shape)
    #
