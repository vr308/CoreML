#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:08:26 2021

@author: vr308
"""
import datetime
import json
import math
import pprint
import time
from dataclasses import dataclass
from typing import Type

import gpflow
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from bayesian_benchmarks import data as uci_datasets
from bayesian_benchmarks.data import Dataset
from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils import ExperimentName

LOGS = f"logs/{datetime.datetime.now().strftime('%b%d')}"
ex = Experiment("uci")
ex.observers.append(FileStorageObserver(f"{LOGS}/runs"))


@ex.config
def config():
    model_type = "gpr"
    kernel_type = "SE"
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    # dataset needs to correspond to the exact name in bayesian_benchmarks.data
    # e.g. Power, Wilson_protein, Wilson_3droad, etc.
    dataset = "Yacht"
    split = 0


def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)


def get_path():
    return f"./{LOGS}/{experiment_name()}"


@ex.capture
def get_data(split, dataset):
    data = get_dataset_class(dataset)(split=split)
    print("DATASET N_TRAIN", len(data.X_train))
    print("DATASET N_TEST", len(data.X_test))
    return data


@ex.capture
def experiment_name(
    date,
    dataset,
    model_type,
    kernel_type,
    split,
):
    return (
        ExperimentName(date)
        .add("model", model_type)
        .add("dataset", dataset)
        .add("split", split)
        .add("kernel", kernel_type)
        .get()
    )


@ex.capture
def experiment_info_dict(
    date,
    dataset,
    model_type,
    kernel_type,
    split,
):
    return dict(
        model=model_type,
        dataset=dataset,
        kernel=kernel_type,
        split=split,
    )


def build_model(data_train, model_type, framework):
    model = gpflow.models.GPR()
    return model


@ex.capture
def build_model(data_train, model_type, kernel_type):
    if kernel_type == "SE":
        kernel = gpflow.kernels.SquaredExponential()
    else:
        raise NotImplementedError

    model = gpflow.models.GPR(data_train, kernel=kernel)
    return model


@ex.capture
def train_model(model):
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(),
        model.trainable_variables,
        options=dict(maxiter=1000, disp=1),
    )


def evaluate_model(model, data_test):
    XT, YT = data_test

    mu, var = model.predict_y(XT)

    d = YT - mu
    log_p = norm.logpdf(YT, loc=mu, scale=var ** 0.5)
    mse = np.average(d ** 2)
    rmse = mse ** 0.5
    nlpd = -np.average(log_p)

    return dict(rmse=rmse, mse=mse, nlpd=nlpd)


@ex.automain
def main(dataset, split):
    experiment_name()

    data = get_data(split)
    data_train = data.X_train, data.Y_train
    data_test = data.X_test, data.Y_test

    # Model
    model = build_model(data_train)
    gpflow.utilities.print_summary(model)

    # Train the puppy
    train_model(model)

    # Evaluation
    experiment_metrics = evaluate_model(model, data_test)
    # merge two dictionaries
    experiment_dict = {**experiment_info_dict(), **experiment_metrics}

    with open(f"{get_path()}_results.json", "w") as fp:
        json.dump(experiment_dict, fp)

    print(experiment_name())
    pprint.pprint(experiment_dict)