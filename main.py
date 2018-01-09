#!/usr/bin/env python

from chainer.datasets import tuple_dataset
from chainer import datasets
from chainer import iterators

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np
from PIL import Image
from chainer import optimizers


class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, 11)
            self.conv1_2 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv2_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.conv2_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.fc2 = L.Linear(None, 2, nobias=True)

    def __call__(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = F.max_pooling_2d(conv1_2, ksize=2, stride=2)
        # conv2_1 = self.conv2_1(pool1)
        # conv2_1 = F.relu(conv2_1)
        # conv2_2 = self.conv2_2(conv2_1)
        # conv2_2 = F.relu(conv2_2)
        # pool2 = F.max_pooling_2d(conv2_2, ksize=2, stride=2)
        # fc1 = self.fc1(pool2)
        fc1 = self.fc1(pool1)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        return fc2


def get_meganes():
    X = []
    y = []
    glassDataset = 500
    soloDataset = 500
    for i in range(1, glassDataset):
        fName = f"./graduate-research/screened-glasses/{i}.jpg"
        n = np.array([np.array(Image.open(fName), 'f') / 255])

        X.append(n)
        y.append(1)
    for i in range(1, soloDataset):
        fName = "./graduate-research/screened-solo/%s.JPG" % (i)
        n = np.array([np.array(Image.open(fName), 'f') / 255])
        X.append(n)
        y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X, y


def trainer_logging_preset(trainer, test_iter, model, gpu):
    """
    よく使うログ設定をまとめてやる
    :param trainer:
    :param test_iter:
    :param model:
    :param gpu:
    :return:
    """
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="cg.dot"))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(entries=[
        'epoch',
        'main/loss',
        'main/accuracy',
        "validation/main/loss",
        "validation/main/accuracy",
        'elapsed_time'
    ]))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))


def create_trainer(train_iter, test_iter, epoch, gpu):
    model = L.Classifier(Model())


    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out="out")

    trainer_logging_preset(trainer, test_iter, model, gpu)
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    X, y = get_meganes()

    d = tuple_dataset.TupleDataset(X, y)
    train, test = datasets.split_dataset_random(d, int(len(d) * 0.8), 0)
    print(train)

    batchsize = 30
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    print("model create")
    trainer = create_trainer(train_iter, test_iter, epoch=args.epoch, gpu=args.gpu)
    updater = trainer.updater
    optimizer = updater.get_optimizer("main")
    model = optimizer.target

    print("run!")
    trainer.run()
    return

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
