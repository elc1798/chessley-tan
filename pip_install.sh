#!/bin/bash

source `pwd`/chessley-venv/bin/activate
pip install --upgrade flask python-chess scikit-learn h5py
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
