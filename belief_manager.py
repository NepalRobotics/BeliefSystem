import collections
import logging

import numpy as np

from IPC.socket_ipc_factory import SocketIPCFactory

from kalman import Kalman
import statistics


logger = logging.getLogger(__name__)


class BeliefManager(object):
  """ Deals with high-level belief system control, including sending and receiving
  data as well as interfacing with the Kalman filter. """
  # Constants for x and y indices in coordinate tuples.
  _X = 0
  _Y = 1

  # Z-score to use when calculating error regions.
  ERROR_REGION_Z_SCORE = 1.96
  # The minimum distance apart (m) two points need to be before we'll use them
  # to calculate a transmitter position.
  MIN_DISTANCE = 5

  # We often store readings from the radio as LOB-strength tuples.
  _LOB = 0
  _STRENGTH = 1

  def __init__(self, autopilot_queue, wireless_queue, radio_queue):
    """ Args:
      autopilot_queue: The queue to use for reading autopilot data.
      wireless_queue: The queue to use for communicating with the WiFi system.
      radio_queue: The queue to use for communicating with the RDF system. """
    self._initialize_member_variables()

    # Initialize connections to the Pixhawk reader and wireless
    # communications handler, as well as the transmitter locator.
    self.__autopilot = autopilot_queue
    self.__wireless = wireless_queue
    self.__radio = radio_queue

    self._fetch_autopilot_data()
    # Initialize the Kalman filter.
    self._filter = Kalman((self._observed_position_x,
                            self._observed_position_y),
                           (self._observed_velocity_x,
                            self._observed_velocity_y))

  def _initialize_member_variables(self):
    """ Initializes essential member variables. This is mostly so that we can
    call this function during testing without calling all of __init__. """
    # Keeps track of past good LOB readings. The list lines up 1-to-1 with the
    # order of LOBs in the state. Each item is a tuple containing the
    # LOB reading, its variance, and the x and y components of the drone's
    # current position when this LOB was accurate. These will never be the
    # current state.
    self._past_lobs = []
    # A record of the state we last used to update _past_lobs.
    self._old_state = None
    # Keeps a record of past states.
    self._past_states = collections.deque()
    # Error regions calculated from our last cycle.
    self._last_error_regions = {}

  def _fetch_autopilot_data(self):
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

      self._observed_position_x = None
      self._observed_position_y = None
      self._observed_velocity_x = None
      self._observed_velocity_y = None

      return

    self._observed_position_x = sensor_data.latitude
    self._observed_position_y = sensor_data.longitude

    self._observed_velocity_x = sensor_data.velocity[self._X]
    self._observed_velocity_y = sensor_data.velocity[self._Y]

    logger.info("Got new sensor data: position: (%f, %f), velocity: (%f, %f)" \
                % (self._observed_position_x, self._observed_position_y,
                   self._observed_velocity_x, self._observed_velocity_y))

  def _fetch_radio_data(self):
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
      nomalized_lob = self._filter.normalize_lobs(signal.lob)

      transmitters.append((normalized_lob, signal.strength))

    return transmitters

  def _associate_lob_readings(self, readings):
    """ Takes a set of LOB readings from the radio system, and associates them
    with known transmitters.
    Args:
      readings: The set of readings to associate. Should be a list of tuples of
      the form (LOB, strength)
    Returns: A dictionary. The keys are indexes of transmitters in the state,
      and the values are the tuple of the associated LOB and signal strength. It
      also returns a list of readings that are asumed to be from new
      transmitters. """

    def is_region_bisected(region, reading):
      """ Takes a region and a line and determines if the line goes through the
      region. Note that this is not a trivial operation, it's at worst O(n) on
      the number of points in the region.
      Args:
        region: The region, defined as a point cloud.
        reading: The reading that we are analyzing.
      Returns:
        True if the line goes through the region, False otherwise. """
      # Convert the reading to a slope and intercept.
      lob = reading[self._LOB]
      slope = np.tan(lob)
      intercept = self._observed_position_y - \
                  self._observed_position_x * slope

      found_lower = False
      found_higher = False
      center_x = self._observed_position_x
      center_y = self._observed_position_y
      for point in region:
        # It can't go backwards through the center.
        relative_pos_x = point[self._X] - center_x
        relative_pos_y = point[self._Y] - center_y
        if ((lob > 0 and lob < np.pi / 2.0) and \
            (relative_pos_x < 0 and relative_pos_y < 0)):
          # If the bearing's in the first quadrant, ignore points in the third
          # quadrant.
          continue
        if ((lob > np.pi / 2.0 and lob < np.pi) and \
            (relative_pos_x > 0 and relative_pos_y < 0)):
          # If the bearing's in the second quadrant, ignore points in the fourth
          # quadrant.
          continue
        if ((lob > np.pi and lob < 3.0 * np.pi / 2) and \
            (relative_pos_x > 0 and relative_pos_y > 0)):
          # If the bearing's in the third quadrant, ignore points in the first
          # quadrant.
          continue
        if ((lob > 3.0 * np.pi / 2 and lob < 2.0 * np.pi) and \
            (relative_pos_x < 0 and relative_pos_y > 0)):
          # If the bearing's in the fourth quadrant, ignore points in the second
          # quadrant.
          continue

        # All we need to prove that the line goes through the region is to find
        # two points that fall on different sides of it.
        line_y = slope * point[self._X] + intercept
        if point[self._Y] > line_y:
          found_higher = True
        elif point[self._Y] < line_y:
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
      # Check whether it goes through each region.
      associated = False
      for transmitter, region in self._last_error_regions.iteritems():
        print "Region for transmitter %d:%s" % (transmitter, region)
        if is_region_bisected(region, reading):
          associated = True
          # Jot down that this LOB probably belongs to this transmitter.
          if associations.get(transmitter):
            # Pick the strongest signal.
            if reading[self._STRENGTH] > \
                associations[transmitter][self._STRENGTH]:
              logger.debug("Dropping weak reading for transmitter %d: %s" % \
                           (transmitter, associations[transmitter]))
              associations[transmitter] = reading
          else:
            associations[transmitter] = reading

      if not associated:
        logger.info("Asuming %s is new transmitter." % (str(reading)))
        new_transmitters.append(reading)

    logger.debug("Associated bearings with transmitters: %s" % (associations))
    return (associations, new_transmitters)

  def _distance_from_strength(self, readings):
    """ Takes a set of transmitter readings along with their strengths and uses
    their strengths to guess how far away they are. This is not a very accurate
    method of guaging transmitter positions, but it is used initially, because
    it works with only one reading.
    Args:
      readings: A set of associated LOB readings, or a list of LOB, strength
      tuples. This flexibility is needed because sometimes we need to run this
      function for already associated readings, and it's convenient to not have
      to convert the dictionary that comes out of _associate_lob_readings() into
      a list of tuples just to run this.
    Returns: Either a dictionary where each key is a transmitter index, and each
      value is a tuple of the X and Y position of the transmitter, or, if
      non-associated readings were provided, a simple list of these tuples. """
    if type(readings) == dict:
      positions = {}
      for transmitter, reading in readings.iteritems():
        positions[transmitter] = \
            self._single_point_strength_distance(self._filter.position(),
                                                 reading[self._LOB],
                                                 reading[self._STRENGTH])
    else:
      positions = []
      for reading in readings:
        location = self._single_point_strength_distance(self._filter.position(),
                                                        reading[self._LOB],
                                                        reading[self._STRENGTH])
        positions.append(location)

    return positions

  def _distance_from_past_states(self, transmitters):
    """ Uses data from previous states as a more accurate method for calculating
    transmitter positions. The way this works is that the drone gets two LOBs on
    the transmitter at two different times. Where those LOBs cross is the
    location of the transmitter.
    transmitters: The indexes of the transmitters in the state to use.
    Returns: A dict of tuples, with each tuple containing the estimated x and y
      position of the transmitter. The keys are the indices of the corresponding
      transmitters in the state. """
    print "Past lobs: Calculating distance."
    positions = {}
    for transmitter in transmitters:
      # Grab the data we need from the filtered state.
      current_lob = self._filter.state()[transmitter]
      current_position = self._filter.position()

      # To find out where they cross, we first have to translate the lobs into
      # actual lines. Start with the current one.
      intersection = self._calculate_intersection(transmitter - Kalman.LOB,
                                                  current_lob, current_position)

      print "Past lobs: intersect %s" % (str(intersection))
      positions[transmitter] = intersection

    return positions

  def _calculate_intersection(self, transmitter, new_lob, position):
    """ Calculates where two LOBs on the same transmitter intersect.
    The new LOB is supplied, and the old LOB is taken from the saved state
    information.
    Args:
      transmitter: The index in the _past_lobs array of the transmitter in
      question.
      new_lob: The new LOB to this transmitter.
      position: The current position of the drone to use in the calculation.
    Returns:
      A tuple containing an x and y coordinate for the point of intersection.
    """
    # To find out where they cross, we first have to translate the lobs into
    # actual lines. Start with the current one.
    current_slope = np.tan(new_lob)
    current_intercept = position[self._Y] - current_slope * \
                        position[self._X]

    # Translate the old one. Each item in past_lobs is organized as a tuple,
    # where the first item is the actual LOB, the second item is the
    # corresponding strength, and the third and fourth items are the x and y
    # components of the corresponding drone position, respectively.
    past_data = self._past_lobs[transmitter]
    past_lob = past_data[0]
    past_x = past_data[2]
    past_y = past_data[3]
    past_slope = np.tan(past_lob)
    past_intercept = past_y - past_slope * past_x

    # Calculate where they intersect.
    intersect_x = (current_intercept - past_intercept) / \
                  (past_slope - current_slope)
    intersect_y = past_slope * intersect_x + past_intercept

    return (intersect_x, intersect_y)

  def _single_point_strength_distance(self, position, lob, strength):
    """ Does the distance-from-strength calculation for a single point.
    Args:
      position: The drone position to use in the calculation. (X, Y)
      lob: The LOB to the transmitter that we are using.
      strength: The strength associated with the LOB reading.
    Returns:
      A tuple containing the X and Y coordinates of the calculated position. """
    # TODO(danielp): Implement this function for real.
    logger.critical("_distance_from_strength not implemented!")

    # For now, strengths have a 1-to-1 correlation with distance.
    position_x = position[self._X] + strength * np.cos(lob)
    position_y = position[self._Y] + strength * np.sin(lob)

    return (position_x, position_y)

  def _calculate_distance(self, readings):
    """ Use the best method to calculate distance, given the circumstances.
    readings: Associated LOB readings.
    Returns:
      A dict of tuples, with each tuple constaining the estimated x and y
      position of the transmitter. The keys are the indices of the corresponding
      transmitters in the state. """
    strength_readings = {}
    past_measurement_transmitters = []
    for transmitter, reading in readings.iteritems():
      current_position = self._filter.position()
      print "Past lobs: transmitter: %d" % (transmitter)
      print "Past lobs: length: %d" % (len(self._past_lobs))
      if transmitter - Kalman.LOB >= len(self._past_lobs):
        # Use strength.
        strength_readings[transmitter] = reading
        continue

      past_x = self._past_lobs[transmitter - Kalman.LOB][2]
      past_y = self._past_lobs[transmitter - Kalman.LOB][3]

      past_measurement_transmitters.append(transmitter)

    distances = self._distance_from_strength(strength_readings)
    print "Cycle: transmitters: %s" % (past_measurement_transmitters)
    distances2 = self._distance_from_past_states(past_measurement_transmitters)
    # Combine them.
    for transmitter, position in distances2.iteritems():
      distances[transmitter] = distances2[transmitter]

    return distances

  def _transmitter_error_regions(self, stddevs, existing, new):
    """ Calculates error regions for the transmitters that we can currently see,
    and saves them.
    Args:
      stddevs: The z-score to use when calculating the error regions.
      existing: Associated readings for existing transmitters.
      new: Readings for new transmitters. """
    # This is kind of a dumb heuristic for calculating the number of points we
    # need to approximate our error ellipsoid. Basically, we use ten points for
    # each dimension, and there is one dimension for each item in the state.
    num_points = 40 + 10 * self._filter.number_of_transmitters()
    error_points = statistics.error_ellipse(self._filter.state_covariances(),
                                            self._filter.state(), stddevs,
                                            num_points)

    # For each transmitter, start building up transformed sets of points that
    # describe the error regions around that particular transmitter. The way
    # this works is that each point on the ellipsoid represents a unique point
    # in state-space which falls within our confidence region. However, since
    # the state is in the form of the drone position and LOBs, this isn't very
    # useful for actually drawing a 2D error region around each transmitter.
    # Therefore, we have to go through and calculate that region from the points
    # in the ellipsoid.
    new_index = 0
    for i in range(Kalman.LOB, len(self._filter.state())):
      if (i - Kalman.LOB >= len(self._past_lobs)):
        # It's new enough that we're going to have to rely on the strength
        # calculations instead.
        print "Calculating error region with strength."
        logger.debug("Falling back on strength calculator for %d." % (i))

        if i >= len(self._filter.state()) - len(new):
          # New transmitter. They were added sequentially, so finding the
          # right one is easy.
          strength = new[new_index][self._STRENGTH]
          new_index += 1
        else:
          # Existing transmitter.
          pair = existing.get(i)
          if not pair:
            # We don't have any data on this transmitter, so our hands are
            # kind of tied.
            logger.warning("Skipping transmitter %d with no data." % (i))
            continue
          strength = pair[self._STRENGTH]

      transformed_points = []
      for point in error_points:
        print "Error point: %s" % (str(point))
        # Each point on the error ellipsoid has the same layout as the state...
        hypothetical_lob = point[i]
        hypothetical_position = (point[self._X], point[self._Y])

        # Re-calculate the transmitter location using the hypothetical lob and
        # drone position from within the error region instead.
        if (i - Kalman.LOB >= len(self._past_lobs)):
          logger.debug("Using strength: %f" % (strength))
          print "Hypothetical pos: %s lob: %f strength: %f" % \
              (str(hypothetical_position), hypothetical_lob, strength)
          new_location = \
              self._single_point_strength_distance(hypothetical_position,
                                                   hypothetical_lob, strength)
          print "Hypothetical new: %s" % (str(new_location))
        else:
          new_location = self._calculate_intersection(i - Kalman.LOB,
                                                      hypothetical_lob,
                                                      hypothetical_position)
        transformed_points.append(new_location)

      self._last_error_regions[i] = transformed_points

    # Now we should have error regions around each transmitter.
    logger.debug("Transmitter error regions: %s\n", self._last_error_regions)

  def _update_past_lobs(self):
    """ Updates the list of past LOBs using new data. """
    current_x = self._filter.position()[self._X]
    current_y = self._filter.position()[self._Y]

    # Find the most recent item in past_states that's far enough away from our
    # current location.
    use_state = None
    use_covariance = None
    used = 0
    for state, covariance in self._past_states:
      old_pos_x = state[self._X]
      old_pos_y = state[self._Y]
      if (np.sqrt((current_x - old_pos_x) ** 2 + \
                  (current_y - old_pos_y) ** 2) >= self.MIN_DISTANCE):
        use_state = state
        use_covariance = covariance
        used += 1
      else:
        # They're just going to get closer as we go up.
        break

    print "Past lobs: use_state: %s" % (str(use_state))

    if use_state == None:
      # We didn't find anything far enough away.
      return

    for i in range(0, used):
      # Once we've used them here, we don't have any use for them anymore.
      self._past_states.popleft()

    for i in range(0, len(use_state) - Kalman.LOB):
      use_lob = use_state[i + Kalman.LOB]
      use_variance = use_covariance[i + Kalman.LOB][i + Kalman.LOB]
      use_x = use_state[self._X]
      use_y = use_state[self._Y]
      save_info = (use_lob, use_variance, use_x, use_y)

      if i >= len(self._past_lobs):
        # This is a new one, add it automatically.
        logger.debug("Saving info %s for new LOB at %d." % (save_info, i))
        self._past_lobs.append(save_info)
        continue

      # Check if it's any more accurate than the one we have currently.
      if use_variance < self._past_lobs[i][1]:
        logger.debug("Updating saved lob from %s to %s." % (self._past_lobs[i],
                                                            save_info))
        self._past_lobs[i] = save_info

    self._old_state = (use_state, use_covariance)

  def iterate(self):
    """ Runs a single iteration of the belief manager. """
    self._fetch_autopilot_data()

    # Check what transmitters we can see.
    readings = self._fetch_radio_data()
    print "Got raw readings: %s" % (readings)
    # Figure out which readings correspond with which transmitters.
    existing, new = self._associate_lob_readings(readings)
    print "Got new readings: %s" % (new)
    print "Got old readings: %s" % (existing)
    # Estimate a position for new transmitters based on the strength.
    new_transmitter_positions = self._distance_from_strength(new)
    # Add new transmitters to the state.
    for i in range(0, len(new)):
      logger.debug("Adding new transmitter at %s with bearing %f." % \
                   (new_transmitter_positions[i], new[i][self._LOB]))
      self._filter.add_transmitter(new[i][self._LOB],
                                   new_transmitter_positions[i])

    # Calculate a position for the old transmitters based on old states.
    existing_transmitter_positions = self._calculate_distance(existing)
    # Update the positions.
    self._filter.set_transmitter_positions(existing_transmitter_positions)

    # Set the measurements.
    # The last few lobs will all be the new ones.
    new_component = [reading[self._LOB] for reading in new]
    # The first lobs will be existing ones or will be masked out if we have no
    # measurement.
    size = self._filter.number_of_transmitters() - len(new_component)
    existing_measurements = [None] * size

    lob_measurements = existing_measurements + new_component
    # Populate the existing LOB measurements.
    for index, reading in existing.iteritems():
      lob_measurements[index - Kalman.LOB] = reading[self._LOB]
    print "LOB measurements: \n%s" % (lob_measurements)

    logger.debug("LOB Measurements: %s" % (lob_measurements))
    self._filter.set_observations((self._observed_position_x,
                                   self._observed_position_y),
                                  (self._observed_velocity_x,
                                   self._observed_velocity_y),
                                  *lob_measurements)

    self._past_states.append((self._filter.state(),
                              self._filter.state_covariances()))
    # Update our selection of past LOBs.
    self._update_past_lobs()
    print "Past lobs: new: %s" % (self._past_lobs)

    self._filter.update()

    # Save new error regions.
    self._transmitter_error_regions(self.ERROR_REGION_Z_SCORE, existing, new)
