""" Tests for belief_manager.py. """


import tests

from belief_manager import BeliefManager
from kalman import Kalman


class TestingBeliefManager(BeliefManager):
  """ A class for testing the belief manager. """
  def __init__(self):
    self.__initialize_member_variables()

    # Zero out initial observations.
    self.__observed_position_x = 0
    self.__observed_position_y = 0
    self.__observed_velocity_x = 0
    self.__observed_velocity_y = 0

    self.__filter = Kalman((self.__observed_position_x,
                            self.__observed_position_y),
                           (self.__observed_velocity_x,
                            self.__observed_velocity_y))

  def set_autopilot_data(self, position, velocity):
    """ Allows us to set the autopilot data that will be fed into the class each
    cycle.
    Args:
      position: Tuple of the x and y position components.
      velocity: Tuple of the x and y velocity components. """
    self.__observed_position_x = position[self._X]
    self.__observed_position_y = position[self._Y]

    self.__observed_velocity_x = velocity[self._X]
    self.__observed_velocity_y = velocity[self._Y]

  def set_radio_data(self, readings):
    """ Allows us to set the radio data that will be fed into the class each
    cycle.
    Args:
      readings: A list of tuples of fake radio readings. Each tuple contains an
      LOB and a strength. """
    self.__radio_data = readings

  def __fetch_autopilot_data(self):
    """ Does nothing, but overrides the actual version of this method, which we
    don't want to run during testing. """
    pass

  def __fetch_radio_data(self):
    """ Overrides the actual version of this method, which we don't want to run
    during testing.
    Returns:
      The radio data last set using set_radio_data. """
    return self.__radio_data


class BeliefManagerTests(tests.BaseTest):
  """ Tests for the BeliefManager class. """
  def setUp(self):
    self.belief
