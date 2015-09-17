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

  """ Tests that the model prediction still works if we rotate the whole thing
  90 degrees. """
  def test_going_forward(self):
    basic_filter = kalman.Kalman((0, 1), (0, 1))
    basic_filter.add_transmitter(0, (0, 5))
    basic_filter.set_observations((0, 2), (0, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(0, state[0], 0.01)
    self._assert_near(2, state[1], 0.01)
    self._assert_near(0, state[2], 0.01)
    self._assert_near(1, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((0, 3), (0, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(0, state[0], 0.01)
    self._assert_near(3, state[1], 0.01)
    self._assert_near(0, state[2], 0.01)
    self._assert_near(1, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)

  """ Tests that the model still works if we go at 45 degrees. """
  def test_going_diagonal(self):
    basic_filter = kalman.Kalman((0, 0), (1, 1))
    basic_filter.add_transmitter(0, (5, 5))
    basic_filter.set_observations((1, 1), (1, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(1, state[0], 0.01)
    self._assert_near(1, state[1], 0.01)
    self._assert_near(1, state[2], 0.01)
    self._assert_near(1, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((2, 2), (1, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(2, state[0], 0.01)
    self._assert_near(2, state[1], 0.01)
    self._assert_near(1, state[2], 0.01)
    self._assert_near(1, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)

  """ Tests that adding new transmitters in the middle works. """
  def test_transmitter_adding(self):
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(0, (5, 0))
    basic_filter.set_observations((2, 0), (1, 0), 0)
    basic_filter.update()

    # Now, add another one.
    basic_filter.add_transmitter(0, (10, 0))
    basic_filter.set_observations((3, 0), (1, 0), 0, 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[0], 0.01)
    self._assert_near(0, state[1], 0.01)
    self._assert_near(1, state[2], 0.01)
    self._assert_near(0, state[3], 0.01)
    self._assert_near(0, state[4], 0.01)

  """ Tests that we can draw a reasonable position error ellipse. """
  def test_position_error_ellipse(self):
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    width, height, angle = basic_filter.position_error_ellipse(1)

    # Because our initial covariance matrix will make the variances for x and y
    # the same, we expect the ellipse to be a cicle.
    self.assertEqual(width, height)
    # Our angle should be pi/2, since all our covariances are zero.
    self.assertEqual(np.pi / 2.0, angle)

    # Now give it an observation in which y is off a lot more than x.
    basic_filter.set_observations((2, 1), (1, 0))
    basic_filter.update()
    width, height, angle = basic_filter.position_error_ellipse(1)
    # Now, the width should be larger than the height. (It's rotated.)
    self.assertGreater(width, height)
