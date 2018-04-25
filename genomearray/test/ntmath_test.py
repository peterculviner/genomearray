# code for local testing of genomearray code on laublab server
import sys, os
import numpy as np
sys.path.append(os.path.relpath("/home/laublab/notebooks/dropbox_link/culviner/repositories/genomearray/"))
import genomearray as ga
from numpy import nan

# test 5' and 3' slope function : rollingslope(input_array, slope_distance, slope_position)
slope_test = np.arange(0,20,2)
slope_test[0:5] = np.arange(0,10)[0:5]
slope_test = np.asarray([slope_test,slope_test])
# known answers for 5' and 3' functionality
expected_5 = np.array([[ 1. ,  2. ,  2.7,  3. ,  2.8,  2. ,  nan,  nan,  nan,  nan],
                       [ nan,  nan,  nan,  nan, -1. , -2. , -2.7, -3. , -2.8, -2. ]])
expected_3 = np.array([[ nan,  nan,  nan,  nan,  1. ,  2. ,  2.7,  3. ,  2.8,  2. ],
                       [-1. , -2. , -2.7, -3. , -2.8, -2. ,  nan,  nan,  nan,  nan]])

def test_rolling_slope_5prime_shape():
    assert ga.ntmath.rollingslope(slope_test, 5, '5_prime').shape == expected_5.shape

def test_rolling_slope_5prime_value():
    np.testing.assert_equal(ga.ntmath.rollingslope(slope_test, 5, '5_prime'), expected_5)

def test_rolling_slope_3prime_shape():
    assert ga.ntmath.rollingslope(slope_test, 5, '3_prime').shape == expected_3.shape

def test_rolling_slope_3prime_value():
    np.testing.assert_equal(ga.ntmath.rollingslope(slope_test, 5, '3_prime'), expected_3)