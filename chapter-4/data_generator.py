from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()


def generate_data():
    X, y = spiral_data(samples=100, classes=3)
    return X, y
