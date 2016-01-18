#!/bin/bash

# Install dependencies for Linux:
sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base
sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install gfortran libopenblas-dev liblapack-dev
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
source `pwd`/chessley-venv/bin/activate
pip install --upgrade --no-use-wheel --no-cache-dir flask python-chess scipy scikit-learn h5py
pip install --upgrade --no-use-wheel --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
