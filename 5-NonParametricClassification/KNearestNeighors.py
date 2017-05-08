#!/usr/bin/python3.6
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(n_samples = 100, n_features = 6, n_classes = 20)
print(X)
