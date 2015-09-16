import logging
import math

import numpy as np

from pykalman import AdditiveUnscentedKalmanFilter


logger = logging.getLogger(__name__)


""" Adds a new row and column of zeroes to a matrix.
Returns: A modified version of the matrix. """
def _expand_matrix(matrix):
  shape = matrix.shape
  new_shape = (shape[0] + 1, shape[1] + 1)

  # Create a matrix of zeros with the new size, and map our old one onto it.
  new_matrix = np.zeros(new_shape)
  new_matrix[:shape[0], :shape[1]] = matrix

  return new_matrix


""" Handles a Kalman filter for computing drone and transmitter locations. """
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
  velocity: How fast we're going. (X, Y) """
  def __init__(self, position, velocity):
    # The transmitter x and y position should technically never change, so it's
    # not part of our state.
    self.__transmitter_positions = []

    # We need an x and y position, an x and y velocity. (LOBs go after that.)
    self.__state = np.array([position[0], position[1], velocity[0], velocity[1]])
    logger.debug("Initial state: %s" % (self.state))
    self.__state_size = self.__state.shape[0]
    # Default observations.
    self.__observations = np.copy(self.__state)

    # Build a covariance matrix. We will use our initial uncertainty constants,
    # and assume that nothing in the state is correlated with anything else.
    self.__state_covariances = np.zeros((self.__state_size, self.__state_size))
    # Calculate our diagonal fill, which will repeat, since the state repeats
    # for each drone.
    diagonal_fill = [self.DRONE_POSITION_UNCERTAINTY,
                     self.DRONE_POSITION_UNCERTAINTY,
                     self.DRONE_VELOCITY_UNCERTAINTY,
                     self.DRONE_VELOCITY_UNCERTAINTY]
    diagonal_indices = np.diag_indices(self.__state_size)
    self.__state_covariances[diagonal_indices] = diagonal_fill
    logger.debug("Initializing state covariances: %s" % \
                 (self.__state_covariances))

    # Since our initial state is directly what's measured from our sensors, it
    # seems logical that the initial observation covariances would be about what
    # our initial state covariances are.
    self.__observation_covariances = np.copy(self.__state_covariances)

    # Initialize the Kalman filter.
    self.__kalman = \
    AdditiveUnscentedKalmanFilter( \
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

    for i in range(0, len(self.__transmitter_positions)):
      position = self.__transmitter_positions[i]
      # We can calculate our LOB too, based on our position.
      new_state[i + 4] = math.atan2(position[1] - current_state[1],
                                    position[0] - current_state[0])

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
  velocity: How fast we think we're going. (X, Y)
  Additional arguments are the LOBs on any transmitters we are tracking. """
  def set_observations(self, position, velocity, *args):
    observations = [position[0], position[1], velocity[0], velocity[1]]

    expecting_lobs = self.__observations.shape[0] - len(observations)
    if len(args) != expecting_lobs:
      raise ValueError("Expected %d LOBs, got %d." % (expecting_lobs,
                                                      len(args)))

    observations.extend(args)
    self.__observations = np.array(observations)
    logger.debug("Setting new observations: %s" % (self.__observations))

  """ Updates the filter for one iteration. """
  def update(self):
    logger.info("Updating kalman filter.")
    output = self.__kalman.filter_update(self.__state, self.__state_covariances,
                                         observation=self.__observations,
                                         observation_covariance=
                                            self.__observation_covariances)
    self.__state, self.__state_covariances = output
    logger.debug("New state: %s, New state covariance: %s" % \
                 (self.__state, self.__state_covariances))

  """ Returns: The current state. """
  def state(self):
    return self.__state

  """ Returns: The current state covariances. """
  def state_covariances(self):
    return self.__state_covariances

  """ Adds a new transmitter for us to track.
  lob: Our LOB to the transmitter.
  location: Where we think that the transmitter is located. """
  def add_transmitter(self, lob, location):
    self.__transmitter_positions.append(location)

    # Add the LOB to the state.
    self.__state = np.append(self.__state, lob)
    new_state_size = self.__state.shape[0]
    # Make sure the observations vector is the proper size.
    self.__observations = np.append(self.__observations, lob)

    # Now resize everything else to make it fit. Start by adding an extra row
    # and column of zeros to the state covariance matrix.
    new_state_cov = _expand_matrix(self.__state_covariances)
    # Add in a new default value for the variance of the new LOB.
    new_state_cov[new_state_size - 1, new_state_size - 1] = \
        self.LOB_UNCERTAINTY

    self.__state_covariances = new_state_cov
    self.__state_size = new_state_size
    logger.debug("New state size: %d" % (self.__state_size))
    logger.debug("New state covariances: %s" % (self.__state_covariances))

    # Do the same thing for the observation covariances.
    new_observation_cov = _expand_matrix(self.__observation_covariances)
    new_observation_cov[new_state_size - 1, new_state_size - 1] = \
        self.LOB_UNCERTAINTY

    self.__observation_covariances = new_observation_cov
    logger.debug("New observation covariances: %s" % \
        (self.__observation_covariances))
