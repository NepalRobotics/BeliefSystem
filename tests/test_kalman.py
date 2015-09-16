""" Tests for the Kalman filter. """


import unittest

import numpy as np

import kalman


""" A superclass for all test cases that defines some useful methods. """
class _BaseTest(unittest.TestCase):
  """ Makes sure that a paremeter is within a cetain amount of something else.
  expected: The value we expected.
  actual: The value we got.
  error: The maximum acceptable deviation between expected and actual. """
  def _assert_near(self, expected, actual, error):
    self.assertLess(abs(expected - actual), error)


""" Tests for the Kalman class. """
class KalmanTests(_BaseTest):
  """ Tests that the filter gives reasonable values under extremely basic
  circumstances. """
  def test_basic(self):
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(0, (5, 0))
    basic_filter.set_observations((2, 0), (1, 0), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(2, state[0], 0.01)
    self._assert_near(0, state[1], 0.01)
    self._assert_near(1, state[2], 0.01)
    self._assert_near(0, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((3, 0), (1, 0), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[0], 0.01)
    self._assert_near(0, state[1], 0.01)
    self._assert_near(1, state[2], 0.01)
    self._assert_near(0, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)
