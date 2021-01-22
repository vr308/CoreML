#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:57:07 2021

@author: vr308

Demo for using sacred in building and fitting a GP model

"""
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('a', interactive=True)

@ex.config
def my_config():
    foo = 42
    bar = 'baz'
    if foo > 40:
        a = 10
    else:
        a = 5

@ex.capture
def some_function(foo, bar=10):
    print(foo, bar)

@ex.capture
def fetch_a(a):
    return a

@ex.main
def my_main(a):
    some_function()     #  1  2   3
    #some_function(1)           #  1  42  'baz'
    some_function(1, bar=12)   #  1  42  12
    print(a)

r=ex.run()