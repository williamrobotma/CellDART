#!/usr/bin/env python3
"""Test TensorFlow CUDA installation."""

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.sysconfig.get_build_info())
