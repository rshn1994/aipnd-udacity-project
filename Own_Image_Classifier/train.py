#!/usr/bin/python3

import argparse
from utils import Util

ap = argparse.ArgumentParser(description='Train')

ap.add_argument('data_dir', default="flowers/")
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--gpu', default=False, action='store_true')
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)

pa = ap.parse_args()

data_dir = pa.data_dir
path = pa.save_dir
learning_rate = pa.learning_rate
architecture = pa.arch
hardware = "gpu" if pa.gpu else "cpu"
num_epochs = pa.epochs
print_instance = 10

print("Loading datasets...")
dataloader_train, dataloader_val, dataloader_test, dataset_train = Util.load_data(data_dir)

print("Setting up model architecture...")
model, criterion, optimizer = Util.model_setup(architecture, learning_rate, hardware)

print("Training model...")
Util.train_network(dataloader_train, dataloader_test, model, criterion, optimizer, num_epochs, print_instance, hardware)

print("Validating testing accuracy...")
Util.testing_acc_check(model, dataloader_test, hardware)

print("Saving model to disk...")
Util.save_checkpoint(model, dataset_train, path)

print("Done!")
