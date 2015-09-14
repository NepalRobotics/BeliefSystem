import logging
import math

import numpy as np

from pykalman import AdditiveUnscentedKalmanFilter

logger = logging.getLogger(__name__)

""" Handles a Kalman filter for computing drone and transmitter locations.
NOTE: This class is designed to integrate data from multiple drones, but to
compute the location of exactly one transmitter. If computation of multiple
transmitter locations is desired, multiple Kalman classes should be used. """
# TODO(danielp): Generalize this for more than one transmitter.
# TODO(danielp): Look into incorporating measurements/state from other drones.
class Kalman:
  # Our initial uncertainty of our drone positions. GPS is not terribly
  # innacurate, so there isn't a ton of uncertainty here.
  DRONE_POSITION_UNCERTAINTY = 0.001
  # Out initial uncertainly of our drone velocities. This is reasonably accurate
  # as well.
  DRONE_VELOCITY_UNCERTAINTY = 0.05
  # Our initial uncertainty on our LOBs.
  # TODO(danielp): I have zero idea of what this should be. Figure that out.
  LOB_UNCERTAINTY = 5 * math.pi / 180

  """ position: Where we are. (X, Y)
  velocity: How fast we're going. (X, Y)
  lob: The LOB to the transmitter we are tracking, in radians.
  transmitter_position: Our best guess for the position of the transmitter.
  (X, Y) """
  def __init__(self, position, velocity, lob, transmitter_position):
    # The transmitter x and y position should technically never change, so it's
    # not part of our state.
    self.transmitter_position = transmitter_position

    # We need an x and y position, an x and y velocity,
    # and an LOB.
    self.state = np.array([position[0], position[1], velocity[0], velocity[1],
                           lob])
    logger.debug("Initial state: %s" % (self.state))
    state_size = self.state.shape[0]
    # Default observations.
    self.observations = np.copy(self.state)

    # Build a covariance matrix. We will use our initial uncertainty constants,
    # and assume that nothing in the state is correlated with anything else.
    self.state_covariances = np.zeros((state_size, state_size))
    # Calculate our diagonal fill, which will repeat, since the state repeats
    # for each drone.
    diagonal_fill = [self.DRONE_POSITION_UNCERTAINTY,
                     self.DRONE_POSITION_UNCERTAINTY,
                     self.DRONE_VELOCITY_UNCERTAINTY,
                     self.DRONE_VELOCITY_UNCERTAINTY,
                     self.LOB_UNCERTAINTY]
    diagonal_indices = np.diag_indices(state_size)
    self.state_covariances[diagonal_indices] = diagonal_fill
    logger.debug("Initializing state covariances: %s" % \
                 (self.state_covariances))

    # Since our initial state is directly what's measured from our sensors, it
    # seems logical that the initial observation covariances would be about what
    # our initial state covariances are.
    self.observation_covariances = np.copy(self.state_covariances)

    # Initialize the Kalman filter.
    self.kalman = \
    AdditiveUnscentedKalmanFilter( \
        observation_covariance=self.observation_covariances,
        transition_functions=self.__transition_function,
        observation_functions=self.__observation_function)

  """ Transition function. Tells us how to get the next state from the current
  state.
  current_state: The current state.
  Returns: The next state. """
  def __transition_function(self, current_state):
    new_state = np.copy(current_state)

    # Updating the position is easy, we just add the velocity.
    new_state[0] += current_state[2]
    new_state[1] += current_state[3]

    # We can calculate our LOB too, based on our position.
    new_state[4] = math.atan2(self.transmitter_position[1] - current_state[1],
                              self.transmitter_position[0] - current_state[0])

    logger.debug("New state prediction: %s" % (new_state))
    return new_state

  """ Observation function. Tells us what our sensor readings should look like
  if our state estimate were correct and our sensors were perfect. """
  def __observation_function(self, current_state):
    # We basically measure our state directly, so this isn't all that
    # interesting.
    return current_state

  """ Sets what our observations are.
  position: Where the GPS thinks we are. (X, Y)
  veloctiy: How fast we think we're going. (X, Y)
  lob: The LOB on the transmitter. """
  def set_observations(self, position, velocity, lob):
    self.observations = np.array([position[0], position[1], velocity[0],
                                  velocity[1], lob])
    logger.debug("Setting new observations: %s" % (self.observations))

  """ Updates the filter for one iteration. """
  def update(self):
    logger.info("Updating kalman filter.")
    output = self.kalman.filter_update(self.state, self.state_covariances,
                                       observation=self.observations)
    self.state, self.state_covariances = output
    logger.debug("New state: %s, New state covariance: %s" % \
                 (self.state, self.state_covariances))
