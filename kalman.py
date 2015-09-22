from operator import add
import logging

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
  LOB_UNCERTAINTY = np.radians(5)

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
      new_state[i + 4] = np.arctan2(position[0] - current_state[0],
                                    position[1] - current_state[1])
      # We use the velocity vector to gauge the drone's heading, and correct the
      # LOB accordingly.
      heading_correction = np.arctan2(current_state[2], current_state[3])
      new_state[i + 4] -= heading_correction

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

  """ Gets a confidence error ellipse for our drone position
  measurement.
  stddevs: How many standard deviations we want the ellipse to encompass.
  Returns: The width, the height, and the angle to the x axis of the ellipse,
  (radians) in a tuple in that order. """
  def position_error_ellipse(self, stddevs):
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

  """ Gets confidence intervals for our LOBs.
  stddevs: How many standard deviations we want the interval to encompass.
  Returns: A list of the margin of errors for each LOB measurement. """
  def lob_confidence_intervals(self, stddevs):
    # Get the indices of the LOB covariances.
    indices = np.diag_indices(self.__state_size)
    # The first four are the position and velocity variances.
    indices = (indices[0][4:], indices[1][4:])
    if not indices[0]:
      # We're not tracking any transmitters:
      return []

    # Get the variances.
    variances = self.__state_covariances[indices]
    omega = np.sqrt(variances)

    margins_of_error = stddevs * omega
    logger.debug("Margins of error for LOBs: %s" % (margins_of_error))
    return margins_of_error

  """ Calculates error ellipses for all the
  transmitter positions. It does this by looking at the worst-case scenario for
  both the error on the drone position and the error on the LOB.
  Returns: A list of data for each transmitter. Each item in the list is itself
  a list of 8 points that define the error region for that transmitter. This
  region is roughly fan shaped, and the points are in clockwise order, starting
  from the bottom left corner. """
  def transmitter_error_region(self, stddevs):
    # Basically, we're taking every point on the ellipse and projecting it
    # through the LOB to reach a new error region, which is sort of fan-shaped.
    # Start by finding both the error regions for our position and lobs.
    ellipse_width, ellipse_height, ellipse_angle = \
        self.position_error_ellipse(stddevs)

    # Turn the error ellipse into a set of points.
    center = (self.__state[0], self.__state[1])
    spread_x = ellipse_width / 2.0
    spread_y = ellipse_height / 2.0
    low_x = (center[0] - spread_x * np.cos(ellipse_angle),
             center[1] - spread_x * np.sin(ellipse_angle))
    high_x = (center[0] + spread_x * np.cos(ellipse_angle),
              center[1] + spread_x * np.sin(ellipse_angle))
    low_y = (center[0] - spread_y * np.sin(ellipse_angle),
             center[1] - spread_y * np.cos(ellipse_angle))
    high_y = (center[0] + spread_y * np.sin(ellipse_angle),
              center[1] + spread_y * np.cos(ellipse_angle))

    lob_errors = self.lob_confidence_intervals(stddevs)
    lobs = self.__state[4:]
    output = []
    for i in range(0, len(lobs)):
      lob = lobs[i]
      lob_error = lob_errors[i]
      transmitter_position = self.__transmitter_positions[i]

      lob_low = lob - lob_error
      lob_high = lob + lob_error
      # Figure out what the transmitter position would be for each of these
      # scenarios.
      radius = np.sqrt((transmitter_position[0] - center[0]) ** 2 + \
                       (transmitter_position[1] - center[1]) ** 2)
      transmitter_low = (radius * np.cos(lob_low), radius * np.sin(lob_low))
      transmitter_high = (radius * np.cos(lob_high), radius * np.sin(lob_high))

      # Calculate points on the error ellipse when centered about all three of
      # these positions.
      recenter_vector_low = (transmitter_low[0] - center[0],
                             transmitter_low[1] - center[1])
      recenter_vector_mean = (transmitter_position[0] - center[0],
                              transmitter_position[1] - center[1])
      recenter_vector_high = (transmitter_high[0] - center[0],
                              transmitter_high[1] - center[1])

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
