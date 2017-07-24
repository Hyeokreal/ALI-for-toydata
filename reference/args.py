import argparse

parser = argparse.ArgumentParser(description='hyper parameters for gan')

# learning rate for Discriminator and Generator
parser.add_argument("-lrd", '--d_lr', type=float, default=0.0000005,
                    help='learning rate of optimizer')
parser.add_argument("-lrg", '--g_lr', type=float, default=0.00000025,
                    help='learning rate of optimizer')


parser.add_argument("-b", "--batch_size", type=int, default=100,
                    help='mini batch size')
parser.add_argument("-i", "--iter", type=int, default=2000000,
                    help='epoch : how many times you will iterate')

# output dir paths
parser.add_argument("-s", "--summary_dir", type=str, default='./logs/',
                    help="path where save summary logs")

# x figs
parser.add_argument("-xfd", "--x_fig_dir", type=str, default='/x_figs/',
                    help="path where save summary logs")

# z figs
parser.add_argument("-zfd", "--z_fig_dir", type=str, default='/z_figs/',
                    help="path where save summary logs")




# Regarding data
parser.add_argument("-m", '--mixtures', type=int, default=25,
                    help="number of mixtures")
parser.add_argument('-n', '--num_dots', type=int, default=500,
                    help="number of dots per one class")


parser.add_argument('-l', '--loss', type=str, default='ls',
                    help="select_loss_from_vanilla_or_ls")
parser.add_argument('-o', '--overwrite', type=bool, default=None,
                    help="Overwrite dara of not")



# Layers
parser.add_argument('-bn', '--batch_norm', type=bool, default=True,
                    help="using_batch_norm")
parser.add_argument('-xd', '--x_dim', type=int, default=2,
                    help="dimension_of_x")
parser.add_argument('-zd', '--z_dim', type=int, default=2,
                    help="dimension_of_z")
parser.add_argument('-G', '--G_layer', type=int, default=400,
                    help="number of first layer of Decoder")
parser.add_argument('-E', '--E_layer', type=int, default=400,
                    help="number of first layer of Encoder")
parser.add_argument('-D', '--D_layer', type=int, default=200,
                    help="number of first layer of Discriminator")
parser.add_argument('-PT','--param_trick',type=str, default='each-connected',
                    help = "parameterizing trick")



# parser.add_argument('-Ga', '--G_activation', type=str, default='relu',
#                     help="select activation function of Decoder")
# parser.add_argument('-Ea', '--E_activation', type=str, default='relu',
#                     help="select activation function of Encoder")
# parser.add_argument('-Da', '--D_activation', type=str, default='relu',
#                     help="select activation function of Discriminator")

parser.add_argument('-N', '--note', type=str, default=None,
                    help="note_for_specify_parameters")

args = parser.parse_args()


def get_args():
    return args


def _print():
    tuples = vars(args).items()
    for x in tuples:
        print(x)


if __name__=="__main__":
    _print()

    print(vars(args).items())