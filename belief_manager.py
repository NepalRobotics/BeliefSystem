import logging

import numpy as np

from IPC.socket_ipc_factory import SocketIPCFactory

from kalman import Kalman


logger = logging.getLogger(__name__)


class BeliefManager(object):
  """ Deals with high-level belief system control, including sending and receiving
  data as well as interfacing with the Kalman filter. """
  # Constants for x and y indices in coordinate tuples.
  _X = 0
  _Y = 0

  # We often store readings from the radio as LOB-strength tuples.
  _LOB = 0
  _STRENGTH = 1

  def __init__(self, autopilot_queue, wireless_queue, radio_queue):
    """ Args:
      autopilot_queue: The queue to use for reading autopilot data.
      wireless_queue: The queue to use for communicating with the WiFi system.
      radio_queue: The queue to use for communicating with the RDF system. """
    # Initialize connections to the Pixhawk reader and wireless
    # communications handler, as well as the transmitter locator.
    self.__autopilot = autopilot_queue
    self.__wireless = wireless_queue
    self.__radio = radio_queue

    # Keeps track of past good LOB readings. The list lines up 1-to-1 with the
    # order of LOBs in the state. Each item is a tuple containing the
    # LOB reading, its variance, and the x and y components of the drone's
    # current position when this LOB was accurate. These will never be the
    # current state.
    self.__past_lobs = []
    # Error regions calculated from our last cycle.
    self.__last_error_regions = {}

    self.__fetch_autopilot_data()
    # Initialize the Kalman filter.
    self.__filter = Kalman((self.__observed_position_x,
                            self.__observed_position_y),
                           (self.__observed_velocity_x,
                            self.__observed_velocity_y))

  def __fetch_autopilot_data(self):
    """ Fetches measurements from the onboard Pixhawk autopilot, and updates the
    member variables for vehicle position and velocity observations accordingly.
    """
    # Read the latest data from the autopilot.
    sensor_data = self.__autopilot.read_front(block=False)
    if not sensor_data:
      # No new sensor data. This is really a problem, because it is designed to
      # handle intermittent radio data, but data from the other sensors should
      # be fairly consistent.
      logger.error("Got no new data from autopilot!")

      self.__observed_position_x = None
      self.__observed_position_y = None
      self.__observed_velocity_x = None
      self.__observed_velocity_y = None

      return

    self.__observed_position_x = sensor_data.latitude
    self.__observed_position_y = sensor_data.longitude

    self.__observed_velocity_x = sensor_data.velocity[self._X]
    self.__observed_velocity_y = sensor_data.velocity[self._Y]

    logger.info("Got new sensor data: position: (%f, %f), velocity: (%f, %f)" \
                % (self.__observed_position_x, self.__observed_position_y,
                   self.__observed_velocity_x, self.__observed_velocity_y))

  def __fetch_radio_data(self):
    """ Fetches measurements from the radio.
    Returns:
        A list containing one tuple for every object sighted by the radar.
        Each tuple contains the LOB measurement and signal strength. """
    transmitters = []

    # Keep reading data until we have no more to read.
    while True:
      signal = self.__radio.read_next(block=False)
      if not signal:
        # No more data.
        break

      # Normalize the LOB, so that the angle will still be correct even if the
      # drone starts flying in a different direction.
      # TODO (danielp): Deal with the fact that the drone could have been flying
      # a different direction when this reading was taken.
      nomalized_lob = self.__filter.normalize_lobs(signal.lob)

      transmitters.append((normalized_lob, signal.strength))

    return transmitters

  def __associate_lob_readings(self, readings):
    """ Takes a set of LOB readings from the radio system, and associates them
    with known transmitters.
    Args:
      readings: The set of readings to associate. Should be a list of tuples of
      the form (LOB, strength)
    Returns: A dictionary. The keys are indexes of transmitters in the state,
      and the values are the tuple of the associated LOB and signal strength. It
      also returns a list of readings that are asumed to be from new
      transmitters. """

    def is_region_bisected(self, region, slope, intercept):
      """ Takes a region and a line and determines if the line goes through the
      region. Note that this is not a trivial operation, it's at worst O(n) on
      the number of points in the region.
      Args:
        region: The region, defined as a point cloud.
        slope: The slope of the line.
        intercept: The intercept of the line.
      Returns:
        True if the line goes through the region, False otherwise. """
      found_lower = False
      found_higher = False
      for point in region:
        # All we need to prove that the line goes through the region is to find
        # two points that fall on different sides of it.
        line_y = slope * point[self._X] + intercept
        if point[self._Y] > line_y:
          found_higher = True
        if point[self._Y] < line_y:
          found_lower = True
        else:
          continue

        if found_higher and found_lower:
          return True

        return False

    # Transmitter positions should not change in an ideal world, so we should be
    # able to make a pretty good guess as to what reading corresponds to what
    # transmitter by what error regions they fall into.
    associations = {}
    new_transmitters = []
    # Check which LOBs pass through which error regions.
    for reading in readings:
      # Convert the reading to a slope and intercept.
      slope = np.sin(reading[self._LOB]) / np.cos(reading[self._LOB])
      intercept = self.__observed_position_y - \
                  self.__observed_position_x * slope

      # Check whether it goes through each region.
      associated = False
      for transmitter, region in self.__last_error_regions.iteritems():
        associated = True

        if is_region_bisected(region, slope, intercept):
          # Jot down that this LOB probably belongs to this transmitter.
          if associations.get("transmitter"):
            # Pick the strongest signal.
            if reading[self._STRENGTH] > \
                associations[transmitter][self._STRENGTH]:
              logger.debug("Dropping weak reading for transmitter %d: %s" % \
                           (transmitter, associations[transmitter]))
              associations[transmitter] = reading
          else:
            associations[transmitter] = reading

      if not associated:
        logger.info("Asuming %s is new transmitter." % reading)
        new_transmitters.append(reading)

    logger.debug("Associated bearings with transmitters: %s" % (associations))
    return (associations, new_transmitters)

  def __distance_from_strength(self, readings):
    """ Takes a set of transmitter readings along with their strengths and uses
    their strengths to guess how far away they are. This is not a very accurate
    method of guaging transmitter positions, but it is used initially, because
    it works with only one reading.
    Args:
      readings: A list of tuples of LOB's and strengths.
    Returns: A list of tuples, with each tuple containing the estimated x and
      y position of the transmitter. """
    # TODO(danielp): Implement this function for real.
    logger.critical("__distance_from_strength not implemented!")

    lobs = np.array([reading[self._LOB] for reading in readings])
    strengths = np.array([reading[self._STRENGTH] for reading in readings])

    # For now, strengths have a 1-to-1 correlation with distance.
    positions_x = strengths * np.cos(lobs)
    positions_y = strengths * np.sin(lobs)

    return zip(positions_x, positions_y)

  def __distance_from_past_states(self, readings):
    """ Uses data from previous states as a more accurate method for calculating
    transmitter positions. The way this works is that the drone gets two LOBs on
    the transmitter at two different times. Where those LOBs cross is the
    location of the transmitter.
    Args:
      readings: A set of associated lob readings from the radio. It will only be
      used when the transmitter is relatively new and there is not enough data
      to calculate its position otherwise. In all other cases, the current
      filtered state will be used instead.
    Returns: A dict of tuples, with each tuple containing the estimated x and y
      position of the transmitter. The keys are the indices of the corresponding
      transmitters in the state. """
    positions = {}
    for transmitter, reading in readings:
      # Check whether we have enough data to use the filtered state.
      if transmitter < len(self.__past_lobs):
        current_lob = self.__filter.state()[transmitter]
      else:
        current_lob = reading[self._LOB]
        logger.info("Using direct measurement for transmitter %d." % \
                    (transmitter))

      # To find out where they cross, we first have to translate the lobs into
      # actual lines. Start with the current one.
      current_position = self.__filter.position()
      intersection = self.__calculate_intersection(transmitter, current_lob,
                                                   current_position)
      positions[transmitter] = intersection

    return positions

  def __calculate_intersection(self, transmitter, new_lob, position):
    """ Calculates where two LOBs on the same transmitter intersect.
    The new LOB is supplied, and the old LOB is taken from the saved state
    information.
    Args:
      transmitter: The index in the state of the transmitter in question.
      new_lob: The new LOB to this transmitter.
      position: The current position of the drone to use in the calculation.
    Returns:
      A tuple containing an x and y coordinate for the point of intersection.
    """
    # To find out where they cross, we first have to translate the lobs into
    # actual lines. Start with the current one.
    current_slope = np.sin(current_lob) / np.cos(current_lob)
    current_intercept = position[self._Y] - current_slope * \
                        position[self._X]

    # Translate the old one. Each item in past_lobs is organized as a tuple,
    # where the first item is the actual LOB, the second item is the
    # corresponding strength, and the third and fourth items are the x and y
    # components of the corresponding drone position, respectively.
    past_data = self.__past_lobs[transmitter]
    past_lob = past_data[0]
    past_x = past_data[2]
    past_y = past_data[3]
    past_slope = np.sin(past_lob) / np.cos(past_lob)
    past_intercept = past_y - past_slope * past_x

    # Calculate where they intersect.
    intersect_x = (current_intercept - past_intercept) / \
                  (current_slope - past_slope)
    intersect_y = past_slope * intersect_x + past_intercept

    return (intersect_x, intersect_y)

  def __transmitter_error_regions(self, stddevs):
    """ Calculates error regions for the transmitters that we can currently see,
    and saves them.
    Args:
      stddevs: The z-score to use when calculating the error regions. """
    # This is kind of a dumb heuristic for calculating the number of points we
    # need to approximate our error ellipsoid. Basically, we use ten points for
    # each dimension, and there is one dimension for each item in the state.
    num_points = 40 + 10 * self.__filter.number_of_transmitters()
    error_points = statistics.error_ellipse(self.__filter.state_covariances(),
                                            self.__filter.state(), stddevs,
                                            num_points)

    # For each transmitter, start building up transformed sets of points that
    # describe the error regions around that particular transmitter. The way
    # this works is that each point on the ellipsoid represents a unique point
    # in state-space which falls within our confidence region. However, since
    # the state is in the form of the drone position and LOBs, this isn't very
    # useful for actually drawing a 2D error region around each transmitter.
    # Therefore, we have to go through and calculate that region from the points
    # in the ellipsoid.
    for i in range(Kalman.LOB, len(self.__filter.state())):
      transformed_points = []
      for point in error_points:
        # Each point on the error ellipsoid has the same layout as the state...
        hypothetical_lob = point[i]
        hypothetical_position = (point[self._X], point[self._Y])

        # Re-calculate the transmitter location using the hypothetical lob and
        # drone position from within the error region instead.
        new_intersection = self.__calculate_intersection(i, hypothetical_lob,
                                                         hypothetical_position)
        transformed_points.append(new_intersection)

      self.__last_error_regions[i] = transformed_points

    # Now we should have error regions around each transmitter.
    logger.debug("Transmitter error regions: %s\n", self.__last_error_regions)

  def __update_past_lobs(self):
    """ Updates the list of past LOBs using new data. """
    past_lobs_size = len(self.__past_lobs)
    for i in range(0, self.__filter.number_of_transmitters()):
      current_lob = self.__filter.state()[i + Kalman.LOB]
      current_variance = self.__filter.state_covariances() \
          [i + Kalman.LOB][i + Kalman.LOB]
      current_x = self.__filter.position()[self._X]
      current_y = self.__filter.position()[self._Y]
      save_info = (current_lob, current_variance, current_x, current_y)

      if i >= past_lobs_size:
        # This is a new one, add it automatically.
        logger.debug("Saving info %s for new LOB at %d." % (save_info, i))
        self.__past_lobs.append(save_info)
        continue

      # Check if it's any more accurate than the one we have currently.
      if current_variance < self.__past_lobs[i][1]:
        logger.debug("Updating saved lob from %s to %s." % (self.__past_lobs[i],
                                                            save_info))
        self.__past_lobs[i] = save_info

  def iterate(self):
    """ Runs a single iteration of the belief manager. """
    self.__fetch_autopilot_data()

    # Check what transmitters we can see.
    readings = self.__fetch_radio_data()
    # Figure out which readings correspond with which transmitters.
    existing, new = self.__associate_lob_readings(readings)
    # Estimate a position for new transmitters based on the strength.
    new_transmitter_positions = self.__distance_from_strength(new)
    # Add new transmitters to the state.
    for i in range(0, len(new)):
      logger.debug("Adding new transmitter at %s with bearing %f." % \
                   (new_transmitter_positions[i], new[i]))
      self.__filter.add_transmitter(self, new[i], new_transmitter_positions[i])

    # Calculate a position for the old transmitters based on old states.
    existing_transmitter_positions = self.__distance_from_past_states(existing)
    # Update the positions.
    self.__filter.set_transmitter_positions(existing_transmitter_positions)

    # Set the measurements.
    # The last few lobs will all be the new ones.
    new = [reading[self._LOB] for reading in new]
    # The first lobs will be existing ones or None, if we have no measurement.
    existing = [None] * \
        (self.__filter.number_of_transmitters() - len(measurments))
    lob_measurements = existing + new
    # Populate the existing LOB measurements.
    for index, reading in existing.iteritems():
      lob_measurements[index - Kalman.LOB] = reading[self._LOB]

    logger.debug("LOB Measurements: %s" % (lob_measurements))
    self.__filter.set_observations((self.__observed_position_x,
                                    self.__observed_position_y),
                                   (self.__observed_velocity_x,
                                    self.__observed_velocity_y),
                                   *lob_measurements)

    self.__filter.update()

    # Save new error regions.
    self.__transmitter_error_regions()
    # Update our selection of past LOBs.
    self.__update_past_lobs()
