#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:58:25 2019

@author: vidhi
"""

from __future__ import print_function

import collections
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions
tf.enable_eager_execution()

try:
  tf.compat.v1.enable_eager_execution()
except ValueError:
  pass