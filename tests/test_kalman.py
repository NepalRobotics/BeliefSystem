""" Tests for the Kalman filter. """


import unittest

import numpy as np

import kalman


class _BaseTest(unittest.TestCase):
  # Constants the indexes of different things in the state.
  _POS_X = 0
  _POS_Y = 1
  _VEL_X = 2
  _VEL_Y = 3
  # Index of the first LOB.
  _LOB = 4

  # Constants for the indices of the x and y components in a coordinate tuple.
  _X = 0
  _Y = 1

  """ A superclass for all test cases that defines some useful methods. """
  def _assert_near(self, expected, actual, error):
    """ Makes sure that a paremeter is within a cetain amount of something else.
    Args:
      expected: The value we expected.
      actual: The value we got.
      error: The maximum acceptable deviation between expected and actual. """
    self.assertLess(abs(expected - actual), error)


class KalmanTests(_BaseTest):
  """ Tests for the Kalman class. """
  def test_basic(self):
    """ Tests that the filter gives reasonable values under extremely basic
    circumstances. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(0, (5, 0))
    basic_filter.set_observations((2, 0), (1, 0), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(2, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((3, 0), (1, 0), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)

  def test_going_forward(self):
    """ Tests that the model prediction still works if we rotate the whole thing
    90 degrees. """
    basic_filter = kalman.Kalman((0, 1), (0, 1))
    basic_filter.add_transmitter(0, (0, 5))
    basic_filter.set_observations((0, 2), (0, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(0, state[self._POS_X], 0.01)
    self._assert_near(2, state[self._POS_Y], 0.01)
    self._assert_near(0, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((0, 3), (0, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(0, state[self._POS_X], 0.01)
    self._assert_near(3, state[self._POS_Y], 0.01)
    self._assert_near(0, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)

  def test_going_diagonal(self):
    """ Tests that the model still works if we go at 45 degrees. """
    basic_filter = kalman.Kalman((0, 0), (1, 1))
    basic_filter.add_transmitter(0, (5, 5))
    basic_filter.set_observations((1, 1), (1, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(1, state[self._POS_X], 0.01)
    self._assert_near(1, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((2, 2), (1, 1), 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(2, state[self._POS_X], 0.01)
    self._assert_near(2, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be even closer this time.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.01)

  def test_transmitter_adding(self):
    """ Tests that adding new transmitters in the middle works. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(0, (5, 0))
    basic_filter.set_observations((2, 0), (1, 0), 0)
    basic_filter.update()

    # Now, add another one.
    basic_filter.add_transmitter(0, (10, 0))
    basic_filter.set_observations((3, 0), (1, 0), 0, 0)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(0, state[self._LOB], 0.01)

  def test_position_error_ellipse(self):
    """ Tests that we can draw a reasonable position error ellipse. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    width, height, angle = basic_filter.position_error_ellipse(1.96)

    # Because our initial covariance matrix will make the variances for x and y
    # the same, we expect the ellipse to be a cicle.
    self.assertEqual(width, height)
    # Our angle should be pi/2, since all our covariances are zero.
    self.assertEqual(np.pi / 2.0, angle)

    # Now give it an observation in which y is off a lot more than x.
    basic_filter.set_observations((2, 1), (1, 0))
    basic_filter.update()
    width, height, angle = basic_filter.position_error_ellipse(1.96)
    # Now, the width should be larger than the height. (It's rotated.)
    self.assertGreater(width, height)

  def test_lob_confidence(self):
    """ Tests that we can compute a confidence interval for the LOB data. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    self.assertEqual([], basic_filter.lob_confidence_intervals(1.96))

    # Now give it a transmitter to track.
    basic_filter.add_transmitter(0, (5, 0))
    # We should have a non-zero margin of error.
    error = basic_filter.lob_confidence_intervals(1.96)
    self.assertGreater(error, 0)

  def test_transmitter_error_region(self):
    """ Tests that we can compute an error region for the transmitter position.
    """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    self.assertEqual([], basic_filter.transmitter_error_region(1.96))

    basic_filter.add_transmitter(0, (5, 0))
    error_region = basic_filter.transmitter_error_region(1.96)

    # We know roughly how the points in the error region should relate to one
    # another, so we can do some basic sanity checking on the output.
    bottom_left, left_middle, top_left, top_middle, top_right, right_middle, \
        bottom_right, bottom_middle = error_region[0]

    self.assertGreater(bottom_left[self._Y], left_middle[self._Y])
    self.assertGreater(top_left[self._Y], left_middle[self._Y])
    self.assertLess(bottom_left[self._X], left_middle[self._X])
    self.assertLess(left_middle[self._X], top_left[self._X])

    self.assertGreater(top_middle[self._X], top_left[self._X])
    self.assertLess(top_right[self._X], top_middle[self._X])
    self._assert_near(top_middle[self._Y], 0, 0.01)
    self.assertGreater(top_middle[self._Y], top_left[self._Y])
    self.assertGreater(top_right[self._Y], top_middle[self._Y])

    self.assertGreater(right_middle[self._Y], top_right[self._Y])
    self.assertGreater(right_middle[self._Y], bottom_right[self._Y])
    self.assertLess(right_middle[self._X], top_right[self._X])
    self.assertLess(bottom_right[self._X], right_middle[self._X])

    self.assertGreater(bottom_middle[self._X], bottom_right[self._X])
    self.assertGreater(bottom_middle[self._X], bottom_left[self._X])
    self._assert_near(bottom_middle[self._Y], 0, 0.01)
    self.assertLess(bottom_middle[self._Y], bottom_right[self._Y])
    self.assertLess(bottom_left[self._Y], bottom_middle[self._Y])
