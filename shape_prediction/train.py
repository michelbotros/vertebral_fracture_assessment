import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def train(args):
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()
    train(args)