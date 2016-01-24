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

def _positivise_covariance(matrix):
  """ Nudges values in a matrix to keep it from becoming non-positive definite
  due to floating point innacuracies.
  These changes are sourced from here:
  http://robotics.stackexchange.com/questions/2000/maintaining-positive-
      definite-property-for-covariance-in-an-unscented-kalman-fil
  Returns:
    A new, modified matrix.
  """
  # "Even out" off-diagonal terms.
  new = 0.5 * matrix + 0.5 * matrix.transpose()
  # Prevent underflow errors.
  new = new + 0.001 * np.identity(new.shape[0])
  return new

# TODO(danielp): Look into incorporating measurements/state from other drones.
class Kalman:
  """ Handles a Kalman filter for computing drone and transmitter locations.
  NOTE: LOB readings come in from the radio as angles relative from the
  nose of the plane, where zero is straight forward. Before they are added to
  the filter, they should be transformed according to the velocity, so
  everything in the state is computed with the positive x axis as zero. """

  # Our initial uncertainty of our drone positions. GPS is not terribly
  # innacurate, so there isn't a ton of uncertainty here.
  DRONE_POSITION_UNCERTAINTY = 0.001
  # Out initial uncertainly of our drone velocities. This is reasonably accurate
  # as well.
  DRONE_VELOCITY_UNCERTAINTY = 0.05
  # Our initial uncertainty on our LOBs.
  # TODO(danielp): I have zero idea of what this should be. Figure that out.
  LOB_UNCERTAINTY = np.radians(5)
  # How good our model is of the drone position.
  POSITION_MODEL_UNCERTAINTY = 0.001
  # How good our model is of the drone velocity.
  VELOCITY_MODEL_UNCERTAINTY = 0.0
  # How good our model is of the LOBs.
  LOB_MODEL_UNCERTAINTY = 0.1
  # Factor by which we scale measured changes to transmitter positions before
  # adding them to the model.
  TRANSMITTER_POSITION_GAIN = 0.05

  # The indices of various elements in the state.
  POS_X = 0
  POS_Y = 1
  VEL_X = 2
  VEL_Y = 3
  # The index of the first LOB.
  LOB = 4

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

    # Set the transition covariance, which is what pykalman calls the Q matrix.
    # In this case, we want the covariances of the position and velocity to be
    # rather small, but those of the LOBs to be larger.
    self.__transition_covariances = np.zeros((self.__state_size,
                                              self.__state_size))
    diagonal_fill = [self.POSITION_MODEL_UNCERTAINTY,
                     self.POSITION_MODEL_UNCERTAINTY,
                     self.VELOCITY_MODEL_UNCERTAINTY,
                     self.VELOCITY_MODEL_UNCERTAINTY]
    diagonal_indices = np.diag_indices(self.__state_size)
    self.__transition_covariances[diagonal_indices] = diagonal_fill

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
    new_state[self.POS_X] += current_state[self.VEL_X]
    new_state[self.POS_Y] += current_state[self.VEL_Y]

    for i in range(0, len(self.__transmitter_positions)):
      position = self.__transmitter_positions[i]
      # We can calculate our LOB too, based on our position.
      new_state[self.LOB + i] = self.__estimate_lob(current_state,
                                                    self.LOB + i)

    logger.debug("New state prediction: %s" % (new_state))
    return new_state

  def __estimate_lob(self, current_state, lob):
    """ Next state estimation for a single LOB.
    Args:
      current_state: The current system state.
      lob: The current index of the LOB.
    Returns:
      The predicted next value of the LOB. """
    position = self.__transmitter_positions[lob - self.LOB]
    return np.arctan2(position[self._Y] - current_state[self.POS_Y],
                      position[self._X] - current_state[self.POS_X])

  def __observation_function(self, current_state):
    """ Observation function. Tells us what our sensor readings should look like
    if our state estimate were correct and our sensors were perfect.
    Returns:
      The sensor readings we would expect. """
    # We basically measure our state directly, so this isn't all that
    # interesting.
    return current_state

  def estimate_next_state(self):
    """ Uses the transition function to estimate the next state based on the
    current one.
    Returns:
      The estimated next state. """
    return self.__transition_function(self.__state)

  def set_observations(self, position, velocity, *args):
    """ Sets what our observations are.
    Args:
      position: Where the GPS thinks we are. (X, Y)
      velocity: How fast we think we're going. (X, Y)
      Additional arguments are the LOBs on any transmitters we are tracking.
      Any of the arguments being None means we're missing a measurement. """
    observations = [position[self._X], position[self._Y], velocity[self._X],
                    velocity[self._Y]]

    # We'll simply use the velocity measurements directly in our state, as drone
    # positional accuracy isn't a huge priority, and they're already pretty
    # decent.
    if velocity[self._X] != None:
      self.__state[self.VEL_X] = velocity[self._X]
    if velocity[self._Y] != None:
      self.__state[self.VEL_Y] = velocity[self._Y]

    expecting_lobs = self.__observations.shape[0] - len(observations)
    if len(args) != expecting_lobs:
      raise ValueError("Expected %d LOBs, got %d." % (expecting_lobs,
                                                      len(args)))

    observations.extend(args)
    mask = []
    for item in observations:
      if item == None:
        mask.append(True)
      else:
        mask.append(False)
    self.__observations = np.ma.array(observations, mask=mask)
    logger.debug("Setting new observations: %s" % (self.__observations))

  def update(self):
    """ Updates the filter for one iteration. """
    logger.info("Updating kalman filter.")
    self.__state_covariances = _positivise_covariance(self.__state_covariances)

    output = self.__kalman.filter_update(self.__state, self.__state_covariances,
                                         observation=self.__observations,
                                         observation_covariance=
                                            self.__observation_covariances,
                                         transition_covariance=
                                            self.__transition_covariances)
    self.__state, self.__state_covariances = output
    logger.debug("New state: %s, New state covariance: %s" % \
                 (self.__state, self.__state_covariances))

  def state(self):
    """
    Returns:
      The current state. """
    return self.__state

  def position(self):
    """
    Returns:
      The position from the current state, in form (X, Y). """
    return (self.__state[self.POS_X], self.__state[self.POS_Y])

  def velocity(self):
    """
    Returns:
      The velocity from the current state, in the form (X, Y). """
    return (self.__state[self.VEL_X], self.__state[self.VEL_Y])

  def state_covariances(self):
    """ Returns: The current state covariances. """
    return self.__state_covariances

  def add_transmitter(self, lob, location):
    """ Adds a new transmitter for us to track.
    Args:
      lob: Our LOB to the transmitter. Note that this value should be normalized
      with normalize_lobs() before being added.
      location: Where we think that the transmitter is located. """
    logger.info("Adding transmitter at %s." % (str(location)))
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

    # Do a similar thing for the transition covariances.
    new_transition_cov = _expand_matrix(self.__transition_covariances)
    new_transition_cov[new_state_size - 1, new_state_size - 1] = \
        self.LOB_MODEL_UNCERTAINTY

    self.__transition_covariances = new_transition_cov
    logger.debug("New transition covariances: %s" % \
        (self.__transition_covariances))

  def remove_transmitter(self, index):
    """ Removes a transmitter that was erroneously added.
    Args:
      index: The index of the transmitter in the state array.
    """
    logger.info("Removing transmitter: %d" % (index))

    self.__transmitter_positions.pop(index - 4)

    # Remove it from the state.
    self.__state = np.delete(self.__state, index)
    self.__state_size = self.__state.shape[0]
    # Make sure the observations vector is the propper size.
    self.__observations = np.delete(self.__observations, index)

    # Resize the state covariance.
    self.__state_covariances = np.delete(self.__state_covariances, index, 0)
    self.__state_covariances = np.delete(self.__state_covariances, index, 1)

    # Resize the observation covariances.
    self.__observation_covariances = \
        np.delete(self.__observation_covariances, index, 0)
    self.__observation_covariances = \
        np.delete(self.__observation_covariances, index, 1)

    # Resize the transition covariances.
    self.__transition_covariances = \
        np.delete(self.__transition_covariances, index, 0)
    self.__transition_covariances = \
        np.delete(self.__transition_covariances, index, 1)

  def set_transmitter_positions(self, positions):
    """ Sets new calculated positions for the transmitters.
    Args:
      positions: A dict of positions. The keys are the indices of transmitters
      in the state. """
    # TODO (danielp): Find a better way of integrating new position
    # measurements.
    for index, position in positions.iteritems():
      old_position = self.__transmitter_positions[index - self.LOB]
      shift_x = position[self._X] - old_position[self._X]
      shift_y = position[self._Y] - old_position[self._Y]
      new_pos = (old_position[self._X] + shift_x * \
                    self.TRANSMITTER_POSITION_GAIN,
                 old_position[self._Y] + shift_y * \
                    self.TRANSMITTER_POSITION_GAIN)
      self.__transmitter_positions[index - self.LOB] = new_pos

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
    indices = (indices[0][self.LOB:], indices[1][self.LOB:])
    if not len(indices[0]):
      # We're not tracking any transmitters:
      return []

    # Get the variances.
    variances = self.__state_covariances[indices]
    omega = np.sqrt(variances)

    margins_of_error = stddevs * omega
    logger.debug("Margins of error for LOBs: %s" % (margins_of_error))
    return margins_of_error

  def normalize_lobs(self, lobs):
    """ Takes a set of LOBs and transforms them according to the direction the
    plane is currently pointing, such that angle zero is the positive x axis,
    like in the unit circle.
    Args:
      lobs: Either a single bearing, or a Numpy array of bearings.
    Returns:
      A numpy array of transformed bearings, or a single one, depending on the
      input. """
    heading_correction = np.arctan2(self.__state[self.VEL_Y],
                                    self.__state[self.VEL_X])
    return lobs + heading_correction

  def number_of_transmitters(self):
    """ Returns:
      The number of transmitters we are currently tracking. """
    return len(self.__transmitter_positions)

  def transmitter_positions(self):
    """ Returns:
      Its current best guess as to the position of the transmitters. """
    return self.__transmitter_positions

  def lobs(self):
    """ Returns:
      The set of LOBs in the state. """
    return self.__state[4:]

  def flip_transmitter(self, transmitter):
    """ Flips a transmitter's LOB by 180 degrees. Also updates the assumed
    transmitter position, taking into account the new information.
    Args:
      transmitter: The index of the transmitter in the state which we are
      flipping. """
    # Flip the LOB.
    lob = self.__state[transmitter]
    lob += np.pi
    # Keep it in range.
    lob %= 2 * np.pi

    logger.debug("Flipping LOB %d to %f." % (transmitter, lob))
    self.__state[transmitter] = lob

    # Now update our calculated position for the transmitter.
    trans_x, trans_y = self.__transmitter_positions[transmitter - self.LOB]
    drone_x, drone_y = self.position()
    d_x = trans_x - drone_x
    d_y = trans_y - drone_y
    new_x = drone_x - d_x
    new_y = drone_y - d_y

    logger.debug("Setting flipped position for %d from (%f, %f) to (%f, %f)." \
                 % (transmitter, trans_x, trans_y, new_x, new_y))
    self.__transmitter_positions[transmitter - self.LOB] = (new_x, new_y)
