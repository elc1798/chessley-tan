#!/bin/bash

source `pwd`/chessley-venv/bin/activate
pip install --upgrade --no-use-wheel --no-cache-dir pip
pip install --upgrade --no-use-wheel --no-cache-dir flask python-chess scipy scikit-learn h5py
pip install --upgrade --no-use-wheel --no-cache-dir passlib pymongo
pip install --upgrade --no-use-wheel --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
pip install --upgrade --no-use-wheel --no-cache-dir gunicorn flask

