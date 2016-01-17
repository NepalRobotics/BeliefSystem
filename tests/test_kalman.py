""" Tests for the Kalman filter. """


import numpy as np

import kalman
import tests


class _BaseTest(tests.BaseTest):
  """ Defines common members for all the test cases in this file. """
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


class KalmanTests(_BaseTest):
  """ Tests for the Kalman class. """
  def test_basic(self):
    """ Tests that the filter gives reasonable values under extremely basic
    circumstances. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (5, 0))
    basic_filter.set_observations((2, 0), (1, 0),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(2, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((3, 0), (1, 0),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be close again.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

  def test_going_forward(self):
    """ Tests that the model prediction still works if we rotate the whole thing
    90 degrees. """
    basic_filter = kalman.Kalman((0, 1), (0, 1))
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (0, 5))
    basic_filter.set_observations((0, 2), (0, 1),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(0, state[self._POS_X], 0.01)
    self._assert_near(2, state[self._POS_Y], 0.01)
    self._assert_near(0, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((0, 3), (0, 1),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(0, state[self._POS_X], 0.01)
    self._assert_near(3, state[self._POS_Y], 0.01)
    self._assert_near(0, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be close again.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

  def test_going_diagonal(self):
    """ Tests that the model still works if we go at 45 degrees. """
    basic_filter = kalman.Kalman((0, 0), (1, 1))
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (5, 5))
    basic_filter.set_observations((1, 1), (1, 1),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    # We should see values that our quite close to our observation since,
    # (surprise, surprise) our observation lines up EXACTLY with our model.
    self._assert_near(1, state[self._POS_X], 0.01)
    self._assert_near(1, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should have gotten close to zero, since our observations
    # match our model so well.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

    # Run another perfect iteration.
    basic_filter.set_observations((2, 2), (1, 1),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(2, state[self._POS_X], 0.01)
    self._assert_near(2, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(1, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

    covariances = basic_filter.state_covariances()

    # Our covariances should be close again.
    for x in np.nditer(covariances):
      self._assert_near(0, x, 0.05)

  def test_transmitter_adding(self):
    """ Tests that adding new transmitters in the middle works. """
    basic_filter = kalman.Kalman((1, 0), (1, 0))
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (5, 0))
    basic_filter.set_observations((2, 0), (1, 0),
        basic_filter.normalize_lobs(0))
    basic_filter.update()

    # Now, add another one.
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (10, 0))
    bearing = basic_filter.normalize_lobs(0)
    basic_filter.set_observations((3, 0), (1, 0), bearing, bearing)
    basic_filter.update()

    state = basic_filter.state()

    self._assert_near(3, state[self._POS_X], 0.01)
    self._assert_near(0, state[self._POS_Y], 0.01)
    self._assert_near(1, state[self._VEL_X], 0.01)
    self._assert_near(0, state[self._VEL_Y], 0.01)
    self._assert_near(basic_filter.normalize_lobs(0), state[self._LOB], 0.01)

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
    basic_filter.add_transmitter(basic_filter.normalize_lobs(0), (5, 0))
    # We should have a non-zero margin of error.
    error = basic_filter.lob_confidence_intervals(1.96)
    self.assertGreater(error, 0)

  def test_flip_transmitter(self):
    """ Tests that the flip_transmitter method works. """
    my_filter = kalman.Kalman((0, 0), (1, 1))

    # Do something really easy first.
    lob = np.pi / 4.0
    my_filter.add_transmitter(lob, (2, 2))
    my_filter.flip_transmitter(4)
    self.assertEqual(5.0 * np.pi / 4.0, my_filter.lobs()[0])

    # Now we should be able to flip it back.
    my_filter.flip_transmitter(4)
    self.assertEqual(np.pi / 4.0, my_filter.lobs()[0])
