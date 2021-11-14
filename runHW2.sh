#!/bin/sh
# Use comments to reduce the number of portions of the assignment run
# Warning! Slow!
python3 ./EnsembleLearning/AdaBoost.py
python3 ./EnsembleLearning/Bagging.py b
python3 ./EnsembleLearning/Bagging.py c
python3 ./EnsembleLearning/RandomForest.py d
python3 ./EnsembleLearning/RandomForest.py e
python3 ./LinearRegression/GradientDescent.py a
python3 ./LinearRegression/GradientDescent.py b
python3 ./LinearRegression/GradientDescent.py c