""" Tests for belief_manager.py. """

from collections import deque

import numpy as np

from belief_manager import BeliefManager
from kalman import Kalman
import statistics
import tests


class _TestingBeliefManager(BeliefManager):
  """ A class for testing the belief manager. """

  def __init__(self, initial_position=(0, 0), initial_velocity=(0, 0)):
    """
    Args:
      initial_position: Optional way to specify an initial position.
      initial_velocity: Optional way to specify an initial velocity. """
    self._initialize_member_variables()

    # Zero out initial observations.
    self._observed_position_x, self._observed_position_y = initial_position
    self._observed_velocity_x, self._observed_velocity_y = initial_velocity

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

  def set_old_state(self, state):
    """ Sets the _old_state member variable.
    Args:
      state: The state to set. """
    self._old_state = state

  def _fetch_data(self):
    """ Does nothing, but overrides the actual version of this method, which we
    don't want to run during testing.
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

  def set_observed_transmitters(self, transmitters):
    """ Sets the _observed_transmitters member variable.
    Args:
      transmitters: The observed transmitters to set.
    """
    self._observed_transmitters = transmitters

  def set_cycle_data(self, cycle_data):
    """ Sets the _cycle_data member variable.
    Args:
      cycle_data: What to change _cycle_data to. """
    self._cycle_data = cycle_data

  def set_cycle(self, cycle):
    """ Sets that _cycle member variable.
    Args:
      cycle: The value of cycle to set. """
    self._cycle = cycle

  def prune_transmitters(self, *args, **kwargs):
    """ Gives us access to _prune_transmitters for testing. See docs on
    _prune_transmitters() for more information. """
    return self._prune_transmitters(*args, **kwargs)

  def set_paired_transmitters(self, paired_transmitters):
    """ Sets the _paired_transmitters member variable.
    Args:
      paired_transmitters: The value to set. """
    self._paired_transmitters = paired_transmitters

  def get_paired_transmitters(self):
    """
    Returns:
      The value of _paired_transmitters. """
    return self._paired_transmitters

  def get_paired_strengths(self):
    """
    Returns:
      The value of _paired_strengths. """
    return self._paired_strengths

  def set_paired_strengths(self, paired_strengths):
    """ Sets the _paired_strengths member variable.
    Args:
      paired_strengths: The value to set. """
    self._paired_strengths = paired_strengths

  def condense_virtual_tranmitters(self, *args, **kwargs):
    """ Gives us access to _condense_virtual_tranmitters for testing. See docs
    on _condense_virtual_tranmitters() for more information. """
    return self._condense_virtual_tranmitters(*args, **kwargs)


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
    self.__lob_stddev = np.radians(5)
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
    distance = self.RADIO_RANGE - (strength * self.RADIO_RANGE)

    x_shift = distance * np.cos(lob)
    y_shift = distance * np.sin(lob)
    print "Shifts: (%f, %f) (%f, %f)" % (lob, strength, x_shift, y_shift)
    x = position[self._X] + x_shift
    y = position[self._Y] + y_shift

    return (x, y)

  def _fetch_data(self):
    """ Fetches simulated data from sensors.
    Returns:
      Simulated data from radio. """
    self._observed_velocity_x = np.random.normal(self.__actual_velocity_x,
                                                 self.__velocity_stddev)
    self._observed_velocity_y = np.random.normal(self.__actual_velocity_y,
                                                 self.__velocity_stddev)

    self._observed_position_x = np.random.normal(self.__actual_position_x,
                                                 self.__position_stddev)
    self._observed_position_y = np.random.normal(self.__actual_position_y,
                                                 self.__position_stddev)

    readings = []
    for transmitter in self.__transmitters:
      # There's a chance that we won't even provide data for this one.
      if (np.random.randint(0, 10) >= 7):
        print "Got random reading failure."
        continue

      # Otherwise, the strength is calculated from our distance to it.
      inside = \
          (transmitter[self._X] - self._filter.position()[self._X]) ** 2 + \
          (transmitter[self._Y] - self._filter.position()[self._Y]) ** 2
      distance = np.sqrt(inside)
      if distance > self.RADIO_RANGE:
        # Too far away to be visible.
        continue
      strength = (self.RADIO_RANGE - distance) / self.RADIO_RANGE
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

  def check_position(self, expected):
    """ Checks that the position in the simulation is within 0.5 meters of an
    expected position.
    Args:
      expected: Expected position. (X, Y)
    Returns:
      True if the check passes, False otherwise. """
    position = self._filter.position()
    if abs(position[self._X] - expected[self._X]) > 1:
      return False
    if abs(position[self._Y] - expected[self._Y]) > 1:
      return False

    return True

  def check_velocity(self, expected):
    """ Checks that the velocity in the simulation is within 0.5 m/s of an
    expected velocity.
    Args:
      expected: Expected velocity. (X, Y) """
    velocity = self._filter.velocity()
    if abs(position[self._X] - expected[self._X]) > 1:
      return False
    if abs(position[self._Y] - expected[self._Y]) > 1:
      return False

    return True

  def check_transmitter_position(self, expected):
    """ Checks that a perceived transmitter position in the simulation is
    within 3 meters of an expected position. No individual transmitter is
    specified, it will check to see if ANY transmitter is near the expected
    position.
    Args:
      expected: The expected position. """
    positions = self._filter.transmitter_positions()
    for position in positions:
      x_in_range = abs(position[self._X] - expected[self._X]) <= 3
      y_in_range = abs(position[self._Y] - expected[self._Y]) <= 3
      if (x_in_range and y_in_range):
        return True

    return False

class BeliefManagerTests(tests.BaseTest):
  """ Tests for the BeliefManager class. """
  def setUp(self):
    self.manager = _TestingBeliefManager()

  def test_associate_lob_readings(self):
    """ Tests that we can associate LOB readings successfully. """
    # We'll start with something really obvious.
    reading1 = (np.pi / 4, 1.0)
    reading2 = (-np.pi / 4, 1.0)

    # Add the first reading as a new transmitter.
    associated, new = self.manager.associate_lob_readings([reading1])
    self.assertEqual({}, associated)
    self.assertEqual([reading1], new)

    self.manager.get_filter().add_transmitter(np.pi / 4, (1, 1))

    # Now the first one should already have been seen.
    associated, new = self.manager.associate_lob_readings([reading1, reading2])

    # One should have been classified as a new one, one shouldn't have.
    self.assertEqual([reading2], new)
    self.assertEqual({4: reading1}, associated)

  def test_duplicate_association(self):
    """ Tests that it handles duplicate associations properly. """
    true_reading = (np.pi / 4, 0.5)
    associated, new = self.manager.associate_lob_readings([true_reading])
    self.assertEqual({}, associated)
    self.assertEqual([true_reading], new)
    self.manager.get_filter().add_transmitter(true_reading[0], (1, 1))

    reading1 = (np.pi / 4, 1.0)
    reading2 = (np.pi / 4.5, 0.5)

    associated, new = self.manager.associate_lob_readings([reading1, reading2])
    # It should ignore the second one since it has a lower strength.
    self.assertEqual([], new)
    self.assertEqual({4: reading1}, associated)

    # ...even if we do it in the other order.
    associated, new = self.manager.associate_lob_readings([reading2, reading1])
    self.assertEqual([], new)
    self.assertEqual({4: reading1}, associated)

  def test_complex_association(self):
    """ Tests a more complicated association case. """
    # Add these two transmitters initially.
    true_readings = [(np.pi / 4, 0.5), (0, 0.5)]
    associated, new = self.manager.associate_lob_readings(true_readings)
    self.assertEqual({}, associated)
    self.assertEqual(true_readings, new)

    self.manager.get_filter().add_transmitter(true_readings[0][0], (5, 5))
    self.manager.get_filter().add_transmitter(true_readings[1][0], (10, 0))

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
    self.assertEqual({4: reading2, 5: reading4}, associated)

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

    self.manager.transmitter_error_regions(1.96, {}, [])

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
    min_distance = BeliefManager.MIN_DISTANCE
    self.manager.set_past_states(deque([((-min_distance, 0, 1, 1, 0),
                                         covariance)]))

    # It should add them if they're not there already.
    self.manager.update_past_lobs()
    lobs = self.manager.get_past_lobs()
    self.assertEqual(len(lobs), 1)
    self.assertEqual(lobs[0][0], 0)

    # It should also replace them if we find one with a better variance.
    self.manager.set_past_lobs([(1, 50, 0, 0)])
    self.manager.set_past_states(deque([((-min_distance, 0, 1, 1, 0),
                                         covariance)]))
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

  def test_remove_bad_transmitter(self):
    """ Tests that we can intelligently remove bad transmitters. """
    number_of_transmitters = deque([1] * self.manager.MAX_INNACTIVE_CYCLES)
    self.manager.set_observed_transmitters(number_of_transmitters)

    readings = [(0, 0.5)]
    self.manager.get_filter().add_transmitter(0, (5, 0))
    self.manager.get_filter().add_transmitter(np.pi / 4.0, (1, 1))

    self.manager.set_cycle_data({4: 9, 5: 0})
    self.manager.set_cycle(10)

    self.manager.prune_transmitters(readings)

    # It should have removed the second transmitter.
    state = self.manager.get_filter().state()
    self.assertEqual(5, len(state))
    self.assertEqual(state[4], 0)

  def test_remove_duplicate_transmitter(self):
    """ Tests that it can intelligently remove duplicate transmitters. """
    number_of_transmitters = deque([1] * self.manager.MAX_INNACTIVE_CYCLES)
    self.manager.set_observed_transmitters(number_of_transmitters)

    readings = [(0, 0.5), (np.pi / 4, 0.5)]
    self.manager.get_filter().add_transmitter(0, (5, 0))
    self.manager.get_filter().add_transmitter(0.1, (5, 0.2))
    self.manager.get_filter().add_transmitter(np.pi / 4.0, (1, 1))

    self.manager.set_cycle_data({4: 9, 5: 8})
    self.manager.set_cycle(10)

    self.manager.prune_transmitters(readings)

    # It should have removed the second transmitter.
    state = self.manager.get_filter().state()
    self.assertEqual(6, len(state))
    self.assertEqual(state[4], 0)
    self.assertEqual(state[5], np.pi / 4)

  def test_lob_choosing(self):
    """ Tests that when we have to choose between the two possibilities for a
    transmitter location, we can do so intelligently and reliably. """
    distance = BeliefManager.MIN_PAIR_ELIMINATION_DISTANCE
    manager = _TestingBeliefManager((0, 0), (distance * 2, distance * 2))

    paired_transmitters = set([4])

    reading_list = []
    dstrength = 1.0 / BeliefManager.MIN_DATA_FOR_REGRESSION
    for i in range(1, BeliefManager.MIN_DATA_FOR_REGRESSION + 1):
      reading_list.append(((0, 0), i * dstrength))
    paired_strengths = {4: reading_list}
    manager.get_filter().add_transmitter(np.pi / 4.0,
                                         (distance * 4, distance * 4))

    manager.set_paired_transmitters(paired_transmitters)
    manager.set_paired_strengths(paired_strengths)

    # Pick some new observed values.
    associated = {4: (np.pi / 4.0, 1.0)}
    # Make us move.
    manager.get_filter().set_observations((30, 30), (30, 30), np.pi / 4.0)
    manager.get_filter().update()

    # Now, try to condense transmitters.
    manager.condense_virtual_tranmitters(associated)

    # Nothing should have really changed in the filter.
    self._assert_near(np.pi / 4.0, manager.get_filter().state()[4], 0.001)
    self.assertEqual(np.pi / 4.0, associated[4][0])
    # However, _paired_transmitters and _paired_strengths should now be empty.
    self.assertEqual(set(), manager.get_paired_transmitters())
    self.assertEqual({}, manager.get_paired_strengths())

  def test_other_lob_choosing(self):
    """ Another test very similar to the one above, except this time it should
    choose the other option. """
    distance = BeliefManager.MIN_PAIR_ELIMINATION_DISTANCE
    manager = _TestingBeliefManager((0, 0), (distance * 2, distance * 2))

    paired_transmitters = set([4])

    reading_list = []
    dstrength = 0.9 / BeliefManager.MIN_DATA_FOR_REGRESSION
    for i in range(1, BeliefManager.MIN_DATA_FOR_REGRESSION + 1):
      reading_list.append(((0, 0), 1.0 - i * dstrength))
    paired_strengths = {4: reading_list}
    manager.get_filter().add_transmitter(np.pi / 4.0,
                                        (distance * 4, distance * 4))

    manager.set_paired_transmitters(paired_transmitters)
    manager.set_paired_strengths(paired_strengths)

    # Pick some new observed values.
    associated = {4: (np.pi / 4.0, 0.1)}
    # Make us move.
    manager.get_filter().set_observations((30, 30), (30, 30), np.pi / 4.0)
    manager.get_filter().update()

    # Now, try to condense transmitters.
    manager.condense_virtual_tranmitters(associated)

    # Our LOB should have flipped.
    self._assert_near(5.0 * np.pi / 4.0, manager.get_filter().state()[4], 0.001)
    self.assertEqual(5.0 * np.pi / 4.0, associated[4][0])
    # Also, _paired_transmitters and _paired_strengths should now be empty.
    self.assertEqual(set(), manager.get_paired_transmitters())
    self.assertEqual({}, manager.get_paired_strengths())

  def test_transmitter_flip_consistency(self):
    """ Tests that it chooses the correct LOB for a transmitter even when we fly
    directly over it, and the LOB from the sensors changes. """
    manager = _TestingBeliefManager((0, 0), (1, 1))
    # Add a transmitter.
    manager.get_filter().add_transmitter(np.pi / 4.0, (2, 2))
    past_states = deque()
    past_states.append(manager.get_filter().state())

    # Make us move.
    manager.get_filter().set_observations((1, 1), (1, 1), np.pi / 4.0)
    manager.get_filter().update()
    manager.set_past_states(past_states)
    past_states.append(manager.get_filter().state())

    # Try to condense transmitters.
    associated = {4: (np.pi / 4.0, 0.5)}
    manager.condense_virtual_tranmitters(associated)

    # Nothing should have changed.
    self.assertEqual(np.pi / 4.0, associated[4][0])

    # Now, move again.
    manager.get_filter().set_observations((2, 2), (1, 1), 0)
    manager.get_filter().update()
    manager.set_past_states(past_states)
    past_states.append(manager.get_filter().state())

    # Try to condense transmitters again.
    associated[4] = (0, 1.0)
    manager.condense_virtual_tranmitters(associated)

    # Again, things should be exactly how we would expect.
    self.assertEqual(0, associated[4][0])

    # Move past the transmitter.
    manager.get_filter().set_observations((3, 3), (1, 1), 5.0 * np.pi / 4.0)
    manager.get_filter().update()
    manager.set_past_states(past_states)
    past_states.append(manager.get_filter().state())

    # Now, the fun part. As soon as we're past the transmitter, the LOB won't
    # flip.
    associated = {4: (np.pi / 4.0, 0.5)}
    manager.condense_virtual_tranmitters(associated)

    # However, the code should be smart enough to recognize that it should.
    self.assertEqual(5.0 * np.pi / 4.0, associated[4][0])

  def test_regression_analysis(self):
    """ Tests that the linear regression analysis throws out bad data. """
    distance = BeliefManager.MIN_PAIR_ELIMINATION_DISTANCE
    manager = _TestingBeliefManager((0, 0), (distance * 2, distance * 2))

    # Here's some bad data. This hardcoded part is here mostly to make sure we
    # don't get a good fit by chance.
    bad_data = [((0, 0), 0.5), ((0, 0), 1.0), ((0, 0), 0.1), ((0, 0), 0.7),
                ((0, 0), 0.3), ((0, 0), 1.0), ((0, 0), 0.8), ((0, 0), 0.1)]
    # Add any extras that are needed.
    for i in range(len(bad_data), BeliefManager.MIN_DATA_FOR_REGRESSION):
      bad_data.append(((0, 0), np.random.random_sample()))

    paired_strengths = {4: bad_data}
    manager.get_filter().add_transmitter(np.pi / 4.0,
                                        (distance * 4, distance * 4))

    paired_transmitters = set([4])
    manager.set_paired_transmitters(paired_transmitters)
    manager.set_paired_strengths(paired_strengths)

    # Pick some new observed values.
    associated = {4: (np.pi / 4.0, 1.0)}
    # Make us move.
    manager.get_filter().set_observations((30, 30), (30, 30), np.pi / 4.0)
    manager.get_filter().update()

    # Now, try to condense transmitters.
    manager.condense_virtual_tranmitters(associated)

    # Nothing should have really changed in the filter.
    self._assert_near(np.pi / 4.0, manager.get_filter().state()[4], 0.001)
    self.assertEqual(np.pi / 4.0, associated[4][0])
    # _paired_transmitters and _paired_strengths should not be empty, because we
    # should not have condensed it.
    self.assertEqual(paired_transmitters, manager.get_paired_transmitters())
    self.assertEqual({4: [bad_data[0]]}, manager.get_paired_strengths())

  def test_iterate_basic(self):
    """ Basically makes sure that iterate() doesn't crash. """
    self.manager.set_radio_data([(0, 1.0)])
    self.manager.iterate()

  def test_iterate_long_term(self):
    """ Simulates an actual use case in order to see how BeliefManager behaves.
    """
    np.random.seed(42)

    # Add a transmitter.
    transmitters = [(50, 50)]
    # Add waypoints that should take us within range of them.
    waypoints = [(70, 20)]

    simulacrum = _EnvironmentSimulator(transmitters, waypoints, (0, 0), 1)

    # Get us away from the test output.
    print ""
    for i in range(0, 73):
      simulacrum.iterate()
      simulacrum.print_report()

    # Check that we ended up in a reasonable place.
    self.assertTrue(simulacrum.check_position(waypoints[0]))
    self.assertTrue(simulacrum.check_transmitter_position(transmitters[0]))

  def test_complex_long_term(self):
    """ Another, more complicated long-term iteration test. """
    np.random.seed(971)

    # Add a few transmitters.
    transmitters = [(50, 50), (20, 25)]
    # Add waypoints that should take us within range of them.
    waypoints = [(70, 20)]

    simulacrum = _EnvironmentSimulator(transmitters, waypoints, (0, 0), 1)

    # Get us away from the test output.
    print ""
    for i in range(0, 73):
      simulacrum.iterate()
      simulacrum.print_report()

    # Check that we ended up in a reasonable place.
    self.assertTrue(simulacrum.check_position(waypoints[0]))
    self.assertTrue(simulacrum.check_transmitter_position(transmitters[0]))
    self.assertTrue(simulacrum.check_transmitter_position(transmitters[1]))
