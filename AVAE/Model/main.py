import os

## VAE Variants
from AVAE import AVAE
#from AVAE import AVAE

from utils import show_all_variables
from utils import check_folder
from shutil import copyfile

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='AVAE',
                        choices=['VAE','AVAE'],  help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='poisson', choices=['mnist', 'poisson'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=45, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=25, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='./Model/checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='./Model/results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./Model/logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    model_dir = './Model/'
    check_folder(model_dir)

    # Make backup copies of files used for model
    copyfile('main.py', model_dir + 'main.py')
    copyfile('utils.py', model_dir + 'utils.py')
    copyfile('ops.py', model_dir + 'ops.py')
    #copyfile('VAE.py', model_dir + 'VAE.py')
    copyfile('AVAE.py', model_dir + 'AVAE.py')
    copyfile('convolution_layers.py', model_dir + 'convolution_layers.py')

    
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    #models = [VAE, AVAE]
    models = [AVAE]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!\n")

        # visualize learned generator
        #gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!\n")

if __name__ == '__main__':
    main()
