from nose.tools import eq_, ok_
from distutils.version import LooseVersion

import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import keras
import tensorflow as tf

def setup_module():
    pass

def test_library_versions():

    # We use Python's distutils to verify versions. It seems sufficient for our purpose.
    # Alternatively, we could use pkg_resources.parse_version for a more robust parse.

    min_python = '3.6'
    max_python = '3.7'
    min_numpy = '1.13'
    min_scipy = '1.0'
    min_keras = '2.0'
    min_tf = '1.9'
    min_pandas = '0.21'
    min_matplotlib = '2.0'

    ok_(LooseVersion(sys.version) > LooseVersion(min_python))
    ok_(LooseVersion(sys.version) < LooseVersion(max_python)) # make sure not to use py3.7
    ok_(LooseVersion(np.__version__) > LooseVersion(min_numpy))
    ok_(LooseVersion(sp.__version__) > LooseVersion(min_scipy))
    ok_(LooseVersion(keras.__version__) > LooseVersion(min_keras))
    ok_(LooseVersion(tf.__version__) > LooseVersion(min_tf))
    ok_(LooseVersion(pd.__version__) > LooseVersion(min_pandas))
    ok_(LooseVersion(matplotlib.__version__) > LooseVersion(min_matplotlib))
