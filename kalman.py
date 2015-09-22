from operator import add
import logging

import numpy as np

from pykalman import AdditiveUnscentedKalmanFilter


logger = logging.getLogger(__name__)


def _expand_matrix(matrix):
  """ Adds a new row and column of zeroes to a matrix.
  Returns:
    A modified version of the matrix. """
  shape = matrix.shape
  new_shape = (shape[0] + 1, shape[1] + 1)

  # Create a matrix of zeros with the new size, and map our old one onto it.
  new_matrix = np.zeros(new_shape)
  new_matrix[:shape[0], :shape[1]] = matrix

  return new_matrix


# TODO(danielp): Look into incorporating measurements/state from other drones.
class Kalman:
  """ Handles a Kalman filter for computing drone and transmitter locations. """
  # Our initial uncertainty of our drone positions. GPS is not terribly
  # innacurate, so there isn't a ton of uncertainty here.
  DRONE_POSITION_UNCERTAINTY = 0.001
  # Out initial uncertainly of our drone velocities. This is reasonably accurate
  # as well.
  DRONE_VELOCITY_UNCERTAINTY = 0.05
  # Our initial uncertainty on our LOBs.
  # TODO(danielp): I have zero idea of what this should be. Figure that out.
  LOB_UNCERTAINTY = np.radians(5)

  # The indices of various elements in the state.
  _POS_X = 0
  _POS_Y = 1
  _VEL_X = 2
  _VEL_Y = 3
  # The index of the first LOB.
  _LOB = 4

  # Indices of x and y in coordinate tuples.
  _X = 0
  _Y = 1

  def __init__(self, position, velocity):
    """
    Args:
      position: Where we are. (X, Y)
      velocity: How fast we're going. (X, Y) """
    # The transmitter x and y position should technically never change, so it's
    # not part of our state.
    self.__transmitter_positions = []

    # We need an x and y position, an x and y velocity. (LOBs go after that.)
    self.__state = np.array([position[self._X], position[self._Y],
                             velocity[self._X], velocity[self._Y]])
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

  def __transition_function(self, current_state):
    """ Transition function. Tells us how to get the next state from the current
    state.
    Args:
      current_state: The current state.
    Returns:
      The next state. """

    new_state = np.copy(current_state)

    # Updating the position is easy, we just add the velocity.
    new_state[self._POS_X] += current_state[self._VEL_X]
    new_state[self._POS_Y] += current_state[self._VEL_Y]

    for i in range(0, len(self.__transmitter_positions)):
      position = self.__transmitter_positions[i]
      # We can calculate our LOB too, based on our position.
      new_state[self._LOB + i] = np.arctan2(position[self._X] - current_state[self._POS_X],
                                    position[self._Y] - current_state[self._POS_Y])
      # We use the velocity vector to gauge the drone's heading, and correct the
      # LOB accordingly.
      heading_correction = np.arctan2(current_state[self._VEL_X],
                                      current_state[self._VEL_Y])
      new_state[self._LOB + i] -= heading_correction

    logger.debug("New state prediction: %s" % (new_state))
    return new_state

  def __observation_function(self, current_state):
    """ Observation function. Tells us what our sensor readings should look like
    if our state estimate were correct and our sensors were perfect.
    Returns:
      The sensor readings we would expect. """
    # We basically measure our state directly, so this isn't all that
    # interesting.
    return current_state

  def set_observations(self, position, velocity, *args):
    """ Sets what our observations are.
    Args:
      position: Where the GPS thinks we are. (X, Y)
      velocity: How fast we think we're going. (X, Y)
      Additional arguments are the LOBs on any transmitters we are tracking. """
    observations = [position[self._X], position[self._Y], velocity[self._X],
                    velocity[self._Y]]

    expecting_lobs = self.__observations.shape[0] - len(observations)
    if len(args) != expecting_lobs:
      raise ValueError("Expected %d LOBs, got %d." % (expecting_lobs,
                                                      len(args)))

    observations.extend(args)
    self.__observations = np.array(observations)
    logger.debug("Setting new observations: %s" % (self.__observations))

  def update(self):
    """ Updates the filter for one iteration. """
    logger.info("Updating kalman filter.")
    output = self.__kalman.filter_update(self.__state, self.__state_covariances,
                                         observation=self.__observations,
                                         observation_covariance=
                                            self.__observation_covariances)
    self.__state, self.__state_covariances = output
    logger.debug("New state: %s, New state covariance: %s" % \
                 (self.__state, self.__state_covariances))

  def state(self):
    """
    Returns:
      The current state. """
    return self.__state

  def state_covariances(self):
    """ Returns: The current state covariances. """
    return self.__state_covariances

  def add_transmitter(self, lob, location):
    """ Adds a new transmitter for us to track.
    Args:
      lob: Our LOB to the transmitter.
      location: Where we think that the transmitter is located. """
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

  def position_error_ellipse(self, stddevs):
    """ Gets a confidence error ellipse for our drone position
    measurement.
    Args:
      stddevs: How many standard deviations we want the ellipse to encompass.
    Returns:
      The width, the height, and the angle to the x axis of the ellipse,
      (radians) in a tuple in that order. """
    # Take the subset of the covariance matrix that pertains to our position.
    position_covariance = self.__state_covariances[:2, :2]

    # Calculate its eigenvalues, and sort them.
    eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ellipse parameters.
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    width, height = 2 * stddevs * np.sqrt(eigenvalues)
    logger.debug("Position error: width: %f, height: %f, angle: %f" % \
        (width, height, angle))

    return (width, height, angle)

  def lob_confidence_intervals(self, stddevs):
    """ Gets confidence intervals for our LOBs.
    Args:
      stddevs: How many standard deviations we want the interval to encompass.
    Returns:
      A list of the margin of errors for each LOB measurement. """
    # Get the indices of the LOB covariances.
    indices = np.diag_indices(self.__state_size)
    # The first four are the position and velocity variances.
    indices = (indices[0][self._LOB:], indices[1][self._LOB:])
    if not indices[0]:
      # We're not tracking any transmitters:
      return []

    # Get the variances.
    variances = self.__state_covariances[indices]
    omega = np.sqrt(variances)

    margins_of_error = stddevs * omega
    logger.debug("Margins of error for LOBs: %s" % (margins_of_error))
    return margins_of_error

  def transmitter_error_region(self, stddevs):
    """ Calculates error ellipses for all the
    transmitter positions. It does this by looking at the worst-case scenario for
    both the error on the drone position and the error on the LOB.
    Returns:
      A list of data for each transmitter. Each item in the list is itself
      a list of 8 points that define the error region for that transmitter. This
      region is roughly fan shaped, and the points are in clockwise order,
      starting from the bottom left corner. """
    # Basically, we're taking every point on the ellipse and projecting it
    # through the LOB to reach a new error region, which is sort of fan-shaped.
    # Start by finding both the error regions for our position and lobs.
    ellipse_width, ellipse_height, ellipse_angle = \
        self.position_error_ellipse(stddevs)

    # Turn the error ellipse into a set of points.
    center = (self.__state[self._POS_X], self.__state[self._POS_Y])
    spread_x = ellipse_width / 2.0
    spread_y = ellipse_height / 2.0
    low_x = (center[self._X] - spread_x * np.cos(ellipse_angle),
             center[self._Y] - spread_x * np.sin(ellipse_angle))
    high_x = (center[self._X] + spread_x * np.cos(ellipse_angle),
              center[self._Y] + spread_x * np.sin(ellipse_angle))
    low_y = (center[self._X] - spread_y * np.sin(ellipse_angle),
             center[self._Y] - spread_y * np.cos(ellipse_angle))
    high_y = (center[self._X] + spread_y * np.sin(ellipse_angle),
              center[self._Y] + spread_y * np.cos(ellipse_angle))

    lob_errors = self.lob_confidence_intervals(stddevs)
    lobs = self.__state[self._LOB:]
    output = []
    for i in range(0, len(lobs)):
      lob = lobs[i]
      lob_error = lob_errors[i]
      transmitter_position = self.__transmitter_positions[i]

      lob_low = lob - lob_error
      lob_high = lob + lob_error
      # Figure out what the transmitter position would be for each of these
      # scenarios.
      radius = np.sqrt((transmitter_position[self._X] - center[self._X]) ** 2 + \
                       (transmitter_position[self._Y] - center[self._Y]) ** 2)
      transmitter_low = (radius * np.cos(lob_low), radius * np.sin(lob_low))
      transmitter_high = (radius * np.cos(lob_high), radius * np.sin(lob_high))

      # Calculate points on the error ellipse when centered about all three of
      # these positions.
      recenter_vector_low = (transmitter_low[self._X] - center[self._X],
                             transmitter_low[self._Y] - center[self._Y])
      recenter_vector_mean = (transmitter_position[self._X] - center[self._X],
                              transmitter_position[self._Y] - center[self._Y])
      recenter_vector_high = (transmitter_high[self._X] - center[self._X],
                              transmitter_high[self._Y] - center[self._Y])

      bottom_left = map(add, low_y, recenter_vector_low)
      left_middle = map(add, low_x, recenter_vector_low)
      top_left = map(add, high_y, recenter_vector_low)
      top_middle = map(add, high_y, recenter_vector_mean)
      top_right = map(add, high_y, recenter_vector_high)
      right_middle = map(add, high_x, recenter_vector_high)
      bottom_right = map(add, low_y, recenter_vector_high)
      bottom_middle = map(add, low_y, recenter_vector_mean)

      # These points define our error region.
      error_region = [bottom_left, left_middle, top_left, top_middle, top_right,
                      right_middle, bottom_right, bottom_middle]
      logger.debug("Error region for transmitter at %s: %s" % \
                   (transmitter_position, error_region))
      output.append(error_region)

    return output
