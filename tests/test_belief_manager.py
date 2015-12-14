""" Tests for belief_manager.py. """

from collections import deque

import numpy as np

from belief_manager import BeliefManager
from kalman import Kalman
import statistics
import tests


class _TestingBeliefManager(BeliefManager):
  """ A class for testing the belief manager. """
  def __init__(self):
    self._initialize_member_variables()

    # Zero out initial observations.
    self._observed_position_x = 0
    self._observed_position_y = 0
    self._observed_velocity_x = 0
    self._observed_velocity_y = 0

    self._filter = Kalman((self._observed_position_x,
                           self._observed_position_y),
                          (self._observed_velocity_x,
                           self._observed_velocity_y))

  def set_autopilot_data(self, position, velocity):
    """ Allows us to set the autopilot data that will be fed into the class each
    cycle.
    Args:
      position: Tuple of the x and y position components.
      velocity: Tuple of the x and y velocity components. """
    self._observed_position_x = position[self._X]
    self._observed_position_y = position[self._Y]

    self._observed_velocity_x = velocity[self._X]
    self._observed_velocity_y = velocity[self._Y]

  def set_radio_data(self, readings):
    """ Allows us to set the radio data that will be fed into the class each
    cycle.
    Args:
      readings: A list of tuples of fake radio readings. Each tuple contains an
      LOB and a strength. """
    self.__radio_data = readings

  def set_error_regions(self, regions):
    """ Allows us to set the past error regions that will be used this cycle.
    Args:
      regions: A dictionary of the points in the region for each transmitter.
      """
    self._last_error_regions = regions

  def set_past_lobs(self, past_lobs):
    """ Allows us to set the content of _past_lobs.
    Args:
      past_lobs: The content for _past_lobs to use. """
    self._past_lobs = past_lobs

  def set_past_states(self, states):
    """ Sets the past states that we have a record of.
    Args:
      states: The past states to set. """
    self._past_states = states

  def set_old_state(self, state):
    """ Sets the _old_state member variable.
    Args:
      state: The state to set. """
    self._old_state = state

  def set_past_states(self, states):
    """ Sets the past states that we have a record of.
    Args:
      states: The past states to set. """
    self._past_states = states

  def set_old_state(self, state):
    """ Sets the _old_state member variable.
    Args:
      state: The state to set. """
    self._old_state = state

  def _fetch_autopilot_data(self):
    """ Does nothing, but overrides the actual version of this method, which we
    don't want to run during testing. """
    pass

  def _fetch_radio_data(self):
    """ Overrides the actual version of this method, which we don't want to run
    during testing.
    Returns:
      The radio data last set using set_radio_data. """
    return self.__radio_data

  def associate_lob_readings(self, *args, **kwargs):
    """ Gives us access to _associate_lob_readings for testing, which is
    otherwise hidden.
    See docs on _associate_lob_readings(). """
    return self._associate_lob_readings(*args, **kwargs)

  def distance_from_past_states(self, *args, **kwargs):
    """ Gives us access to _distance_from_past_states for testing, which is
    otherwise hidden.
    See docs on _distance_from_past_states(). """
    return self._distance_from_past_states(*args, **kwargs)

  def transmitter_error_regions(self, *args, **kwargs):
    """ Gives us access to _transmitter_error_regions for testing, which is
    otherwise hidden.
    See docs on _transmitter_error_regions() for more information. """
    return self._transmitter_error_regions(*args, **kwargs)

  def update_past_lobs(self, *args, **kwargs):
    """ Gives us access to _update_past_lobs for testing, which is otherwise
    hidden.
    See docs on _update_past_lobs() for more information. """
    return self._update_past_lobs(*args, **kwargs)

  def get_filter(self):
    """ Gives us access to the manager filter parameter.
    Returns:
      The filter object for this manager. """
    return self._filter

  def get_last_error_regions(self):
    """ Gives us access to the manager _last_error_regions parameter.
    Returns:
      The value of _last_error_regions. """
    return self._last_error_regions

  def get_past_lobs(self):
    """ Gives us access to the manager _past_lobs parameter.
    Returns:
      The value of _past_lobs. """
    return self._past_lobs


class _EnvironmentSimulator(BeliefManager):
  """ Feeds data to a BeliefManager instance in order to simulate actual
  operation. This class is meant to be used as a replacement for BeliefManager.
  """
  def __init__(self, transmitters, waypoints, position, speed):
    """
    Args:
      transmitters: A list of tuples, each one cotaining an X and Y coordinate
      for a transmitter position. These are the actual locations of the
      simulated transmitters.
      waypoints: The set of points making up the path for the simulated drone to
      follow.
      position: The drone's actual starting position.
      speed: The target ground speed for the simulated drone. """
    self._initialize_member_variables()

    # Zero out initial observations.
    self._observed_position_x = 0
    self._observed_position_y = 0
    self._observed_velocity_x = 0
    self._observed_velocity_y = 0

    self._filter = Kalman((self._observed_position_x,
                           self._observed_position_y),
                          (self._observed_velocity_x,
                           self._observed_velocity_y))

    # These are the parameters that will be used when adding noise to the
    # simulated measurements.
    self.__position_stddev = 0.005
    self.__velocity_stddev = 0.1
    self.__lob_stddev = 0.05
    self.__lob_strength_stddev = 0.1

    # These are the actual locations of the simulated transmitters.
    self.__transmitters = transmitters
    self.__waypoints = waypoints
    self.__current_waypoint = 0

    # The actual values of the drone state.
    self.__actual_position_x = position[self._X]
    self.__actual_position_y = position[self._Y]

    self.__speed = float(speed)
    self.__velocity_for_waypoint(self.__waypoints[self.__current_waypoint])

    # How many iterations we've run
    self.__cycles = 0

    self.__last_x_error = 0
    self.__last_y_error = 0
    self.__passed_x = False
    self.__passed_y = False

  def __velocity_for_waypoint(self, waypoint):
    """ Updates the current velocity so that we are heading towards the
    specified waypoint.
    Args:
      waypoint: The wapoint we are aiming for. """
    target_angle_tan = \
        (float(waypoint[self._Y]) - self._filter.position()[self._Y]) / \
        (float(waypoint[self._X]) - self._filter.position()[self._X])
    # Our goal here is to adjust the angle without changing the magnitude.
    self.__actual_velocity_x = self.__speed / \
        (np.sqrt(1 + target_angle_tan ** 2))
    self.__actual_velocity_y = target_angle_tan * self.__actual_velocity_x

  def __update_velocity_from_waypoints(self):
    """ Updates the drone's velocity in the simulation so that it is heading
    towards the current waypoint. This is basically an approximation of what the
    navigation code will probably do. """
    # Check if we've reached the waypoint.
    waypoint = self.__waypoints[self.__current_waypoint]

    # If we passed our waypoint in both components, it's time to switch.
    x_error = waypoint[self._X] - self._filter.position()[self._X]
    y_error = waypoint[self._Y] - self._filter.position()[self._Y]
    if (x_error * self.__last_x_error < 0):
      # It flipped.
      self.__passed_x = True
    if (y_error * self.__last_y_error < 0):
      self.__passed_y = True

    if (self.__passed_y and self.__passed_y):
      # Go on to the next one.
      self.__current_waypoint += 1
      waypoint = self.__waypoints[self.__current_waypoint]

      self.__passed_x = False
      self.__passed_y = False

    self.__last_x_error = x_error
    self.__last_y_error = y_error

    # Update the velocity so we're heading towards the waypoint.
    self.__velocity_for_waypoint(waypoint)

  def __update_actual_state(self):
    """ Updates the actual state of the simulated drone. """
    self.__update_velocity_from_waypoints()

    self.__actual_position_x += self.__actual_velocity_x
    self.__actual_position_y += self.__actual_velocity_y

  def _single_point_strength_distance(self, position, lob, strength):
    """ Overrides the current function for calculating distances based on
    strengths, since we don't have one. Basically all it does is reverse the
    calculation we used to obtain the strength.
    See the docs for the base class version of _single_point_strength_distance.
    """
    # TODO (danielp): Remove this method once we implement the base class
    # version.
    distance = 40 - (strength * 40)

    x_shift = distance * np.cos(lob)
    y_shift = distance * np.sin(lob)
    print "Shifts: (%f, %f) (%f, %f)" % (lob, strength, x_shift, y_shift)
    x = position[self._X] + x_shift
    y = position[self._Y] + y_shift

    return (x, y)

  def _fetch_autopilot_data(self):
    """ Fetches simulated data from the autopilot. """
    self._observed_velocity_x = np.random.normal(self.__actual_velocity_x,
                                                 self.__velocity_stddev)
    self._observed_velocity_y = np.random.normal(self.__actual_velocity_y,
                                                 self.__velocity_stddev)

    self._observed_position_x = np.random.normal(self.__actual_position_x,
                                                 self.__position_stddev)
    self._observed_position_y = np.random.normal(self.__actual_position_y,
                                                 self.__position_stddev)

  def _fetch_radio_data(self):
    """ Fetches simulated data from the radio system.
    See docs for BeliefManager._fetch_radio_data(). """
    readings = []
    for transmitter in self.__transmitters:
      # There's a chance that we won't even provide data for this one.
      # TODO(danielp): Make it support this.
      #if (np.random.randint(0, 10) >= 7):
      #  print "Got random reading failure."
      #  continue

      # Otherwise, the strength is calculated from our distance to it.
      inside = \
          (transmitter[self._X] - self._filter.position()[self._X]) ** 2 + \
          (transmitter[self._Y] - self._filter.position()[self._Y]) ** 2
      distance = np.sqrt(inside)
      # Assume our range is 40 m.
      if distance > 40:
        # Too far away to be visible.
        continue
      strength = (40 - distance) / 40.0
      strength = np.random.normal(strength, self.__lob_strength_stddev)
      strength = min(strength, 1.0)
      strength = max(0, strength)

      # Calculate the LOB.
      lob = np.arctan2(transmitter[1] - self._observed_position_y,
                       transmitter[0] - self._observed_position_x)
      lob = np.random.normal(lob, self.__lob_stddev)

      readings.append((lob, strength))

    return readings

  def print_report(self):
    """ Prints a little report for the cycle. """
    print "================== CYCLE %d ==================" % (self.__cycles)
    print "Actual State:"
    print "Position: (%f, %f)" % (self.__actual_position_x,
                                  self.__actual_position_y)
    print "Velocity: (%f, %f)" % (self.__actual_velocity_x,
                                  self.__actual_velocity_y)
    print "Current Waypoint: %s" % \
        (str(self.__waypoints[self.__current_waypoint]))
    print "Transmitter Locations: %s" % (self.__transmitters)
    print "----------------------------------------------"

    print "Inferred State:"
    print "Position: %s" % (str(self._filter.position()))
    print "Velocity: %s" % (str(self._filter.velocity()))
    print "LOBs: %s" % (self._filter.state()[4:])
    print "Transmitter Locations: %s" % (self._filter.transmitter_positions())
    print "Covariances: \n%s" % (self._filter.state_covariances())

  def iterate(self):
    """ Runs the iterate() function of the base class, but also updates the
    simulation. """
    self.__update_actual_state()

    super(_EnvironmentSimulator, self).iterate()

    self.__cycles += 1


class BeliefManagerTests(tests.BaseTest):
  """ Tests for the BeliefManager class. """
  def setUp(self):
    self.manager = _TestingBeliefManager()

  def test_associate_lob_readings(self):
    """ Tests that we can associate LOB readings successfully. """
    # We'll start with something really obvious.
    region = [(0, 0), (0, 2), (2, 2), (2, 0)]
    self.manager.set_error_regions({1: region})
    reading1 = (np.pi / 4, 1.0)
    reading2 = (-np.pi / 4, 1.0)

    associated, new = self.manager.associate_lob_readings([reading1, reading2])

    # One should have been classified as a new one, one shouldn't have.
    self.assertEqual([reading2], new)
    self.assertEqual({1: reading1}, associated)

  def test_duplicate_association(self):
    """ Tests that it handles duplicate associations properly. """
    region = [(0, 0), (0, 2), (2, 2), (2, 0)]
    self.manager.set_error_regions({1: region})
    reading1 = (np.pi / 4, 1.0)
    reading2 = (np.pi / 6, 0.5)

    associated, new = self.manager.associate_lob_readings([reading1, reading2])

    # It should ignore the second one since it has a lower strength.
    self.assertEqual([], new)
    self.assertEqual({1: reading1}, associated)

  def test_complex_association(self):
    """ Tests a more complicated association case. """
    test_covariance = np.array([[0.5, 0], [0, 1]])
    region1 = statistics.error_ellipse(test_covariance, (5, 5), 1.96, 50)
    region2 = statistics.error_ellipse(test_covariance, (10, 0), 1.96, 50)
    self.manager.set_error_regions({1: region1, 2: region2})

    # Should go through region 1.
    reading1 = (np.pi / 4, 0.5)
    # Should also go through region 1.
    reading2 = (np.pi / 4.5, 1.0)
    # Should not go through any region.
    reading3 = (-5 * np.pi / 4, 1.0)
    # Should go through region 2.
    reading4 = (0, 1.0)
    readings = [reading1, reading2, reading3, reading4]

    associated, new = self.manager.associate_lob_readings(readings)

    self.assertEqual([reading3], new)

  def test_distance_from_past_states(self):
    """ Basic tests for its ability to calculate distances using information
    from past states. """
    past_lob = (np.pi / 4, 0.1, 0, -5)
    self.manager.set_past_lobs([past_lob])

    current_lob = -np.pi / 4
    self.manager.get_filter().add_transmitter(current_lob, (0, 0))

    positions = self.manager.distance_from_past_states([4])
    self._assert_near(positions[4][0], 2.5, 0.0001)
    self._assert_near(positions[4][1], -2.5, 0.0001)

  def test_transmitter_error_regions(self):
    """ Tests for the _transmitter_error_regions method. """
    self.manager.get_filter().add_transmitter(-np.pi / 4, (0, -2.5))
    self.manager.set_past_lobs([(np.pi / 4, 0.1, 0, -5)])
    self.manager.set_old_state((self.manager.get_filter().state(),
        self.manager.get_filter().state_covariances()))

    self.manager.transmitter_error_regions(1.96)

    points = self.manager.get_last_error_regions()

    # Make sure that our region makes sense.
    for point in points[4]:
      # In the right quadrant.
      self.assertGreater(point[0], 0)
      self.assertLess(point[1], 0)

  def test_update_past_lobs(self):
    """ Tests for the _update_past_lobs method. """
    self.manager.get_filter().add_transmitter(0, (0, 5))
    covariance = self.manager.get_filter().state_covariances()
    self.manager.set_past_states(deque([((-5, 0, 1, 1, 0), covariance)]))

    # It should add them if they're not there already.
    self.manager.update_past_lobs()
    lobs = self.manager.get_past_lobs()
    self.assertEqual(len(lobs), 1)
    self.assertEqual(lobs[0][0], 0)

    # It should also replace them if we find one with a better variance.
    self.manager.set_past_lobs([(1, 50, 0, 0)])
    self.manager.set_past_states(deque([((-5, 0, 1, 1, 0), covariance)]))
    self.manager.update_past_lobs()
    lobs = self.manager.get_past_lobs()
    self.assertEqual(len(lobs), 1)
    self.assertEqual(lobs[0][0], 0)
    self.assertNotEqual(lobs[0][1], 50)

    # Additionally, if we don't have any valid state information that's far
    # enough away from our current position, it shouldn't update saved lobs.
    self.manager.set_past_states(deque([((0, 0, 1, 1, 1), covariance)]))
    self.manager.update_past_lobs()
    lobs = self.manager.get_past_lobs()
    self.assertEqual(len(lobs), 1)
    self.assertEqual(lobs[0][0], 0)

  def test_iterate_no_data(self):
    """ Test that iterate doesn't crash even when we get no data from the
    autopilot. """
    self.manager.set_autopilot_data((None, None), (None, None))
    self.manager.set_radio_data([])
    self.manager.iterate()

  def test_iterate_basic(self):
    """ Basically makes sure that iterate() doesn't crash. """
    self.manager.set_radio_data([(0, 1.0)])
    self.manager.iterate()

  def test_iterate_long_term(self):
    """ Simulates an actual use case in order to see how BeliefManager behaves.
    """
    np.random.seed(42)

    # Add a few transmitters.
    transmitters = [(50, 50)]
    # Add waypoints that should take us within range of them.
    waypoints = [(70, 20)]

    simulacrum = _EnvironmentSimulator(transmitters, waypoints, (0, 0), 1)

    # Get us away from the test output.
    print ""
    for i in range(0, 76):
      simulacrum.iterate()
      simulacrum.print_report()
