import collections
import logging
import sys

from scipy.spatial.distance import euclidean
from scipy.stats import linregress
import numpy as np

from Utils.process import Process
from Utils import nr_logger

from kalman import Kalman
import statistics


logger = logging.getLogger(__name__)


class BeliefManager(Process):
  """ Deals with high-level belief system control, including sending and receiving
  data as well as interfacing with the Kalman filter. """
  # Constants for x and y indices in coordinate tuples.
  _X = 0
  _Y = 1

  # Z-score to use when calculating error regions.
  ERROR_REGION_Z_SCORE = 2.576
  # The minimum distance apart (m) two points need to be before we'll use them
  # to calculate a transmitter position.
  MIN_DISTANCE = 15
  # The number of cycles we have to go without receiving a reading from a
  # particular transmitter in order to deem that transmitter nonexistent.
  MAX_INNACTIVE_CYCLES = 10
  # How far away (m) we can be from a transmitter signal and still register it,
  # in average conditions.
  RADIO_RANGE = 40.0
  # Length of one cycle of the belief manager, in s.
  CYCLE_LENGTH = 0.1
  # How far we have to have traveled (m) in the direction of a transmitter's LOB
  # before we can test it to see whether it needs to flip 180 degrees.
  MIN_PAIR_ELIMINATION_DISTANCE = 15
  # The minimum absolute value of the difference in signal strength at the two
  # positions we are testing before we can make a conclusion about whether the
  # trasmitter has to flip 180 degrees.
  MIN_STRENGTH_DIFF = 0.3
  # The minimum amount of saved data points we have to have to do a linear
  # regression for strength data.
  MIN_DATA_FOR_REGRESSION = 12
  # Minimum value for R^2 in our strength regression before we throw out our
  # data and start over.
  MIN_R_SQUARED = 0.6

  # We often store readings from the radio as LOB-strength tuples.
  _LOB = 0
  _STRENGTH = 1

  def __init__(self, belief_queue, wireless_queue):
    """ Args:
      belief_queue: The queue for reading autopilot and radio data.
      wireless_queue: The queue to use for communicating with the WiFi system.
    """
    super(BeliefManager, self).__init__()

    self._initialize_member_variables()

    # Initialize connections to the wireless communications handler, as well as
    # incomming data from the aggregator process.
    self.__aggregator = belief_queue
    self.__wireless = wireless_queue

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
    # A record of the number of transmitters we've seen for the past few
    # cycles.
    self._observed_transmitters = collections.deque()
    # Error regions calculated from our last cycle.
    self._last_error_regions = {}
    # A dictionary of transmitter indices. Each value is the cycle we last got
    # data for it.
    self._cycle_data = {}
    # A set that keeps track of transmitters for which we still do not know
    # which of the two possible locations for it are correct. Transmitters are
    # stored by their indices in the state.
    self._paired_transmitters = set()
    # For the same transmitters in _paired_transmitters, this dictionary
    # associates the last drone position when the transmitter was sighted, along
    # with the strength, with each transmitter index. It keeps track of a series
    # of such associations for each transmitter.
    self._paired_strengths = {}

    # How many cycles we've run.
    self._cycles = 0

  def _fetch_data(self):
    """ Fetches measurements from the aggregator, updates observed position and
    velocity member variables, and returns radio data.
    Returns:
        A list containing one tuple for every object sighted by the radio.
        Each tuple contains the LOB measurement and signal strength.
    """
    # Read the latest data from the aggregator.
    sensor_data = self.__aggregator.get()

    self._observed_position_x = sensor_data.x_pos
    self._observed_position_y = sensor_data.y_pos

    self._observed_velocity_x = sensor_data.x_velocity
    self._observed_velocity_y = sensor_data.y_velocity

    # We convert them to strings, because they could be None.
    logger.debug("Got new sensor data: position: (%s, %s), velocity: (%s, %s)" \
                % (str(self._observed_position_x),
                   str(self._observed_position_y),
                   str(self._observed_velocity_x),
                   str(self._observed_velocity_y)))

    normalized = []
    for lob, strength in sensor_data.radio_data:
      # Normalize the LOB, so that the angle will still be correct even if the
      # drone starts flying in a different direction.
      # TODO (danielp): Deal with the fact that the drone could have been flying
      # a different direction when this reading was taken.
      nomalized_lob = self._filter.normalize_lobs(lob)

      normalized.append((normalized_lob, strength))

    return normalized

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
    margins_of_error = \
        self._filter.lob_confidence_intervals(self.ERROR_REGION_Z_SCORE)

    # Center the interval around our predicted next LOBs.
    predicted_state = self._filter.estimate_next_state()
    intervals = []
    for i in range(Kalman.LOB, len(predicted_state)):
      lob = predicted_state[i]
      margin_of_error = margins_of_error[i - Kalman.LOB]
      intervals.append((lob - margin_of_error, lob + margin_of_error))

    logger.debug("LOB confidence intervals: %s" % (intervals))

    # Make sure that any paired strengths entries that we don't have data for
    # will end up being set to None.
    for key in self._paired_strengths.keys():
      self._paired_strengths[key].append(None)

    # Now, go and check whether our new LOBs fit within them.
    associations = {}
    new_transmitters = []
    for reading in readings:
      associated = False
      weak = False
      best_center_distance = sys.maxint
      best_transmitter = None

      lob = reading[self._LOB]
      strength = reading[self._STRENGTH]

      for i in range(0, len(intervals)):
        transmitter = i + Kalman.LOB
        lower_bound = intervals[i][0]
        upper_bound = intervals[i][1]
        if (lob >= lower_bound and lob <= upper_bound):
          # It's in range.
          old_association = associations.get(transmitter)
          if old_association:
            # If this one is a stronger signal, replace it.
            if (strength <= old_association[self._STRENGTH]):
              weak = True
              continue
            logger.debug("Replacing weak signal %f with %f." %
                (old_association[self._LOB], lob))

          # Find this reading's distance from the center of the interval.
          center_distance = lob - (lower_bound + \
            (upper_bound - lower_bound) / 2)
          center_distance = abs(center_distance)
          # If it's already associated, we go with whatever reading is closest
          # to our expected value.
          if associated:
            if center_distance >= best_center_distance:
              # Use the other one.
              continue

          associated = True
          best_center_distance = center_distance
          best_transmitter = transmitter

      if associated:
        associations[best_transmitter] = reading
        self._cycle_data[best_transmitter] = self._cycles
        self._paired_strengths[best_transmitter][-1] = \
            (self._filter.position(), reading[self._STRENGTH])

      if (not associated and not weak):
        # It fit inside none of our previous regions.
        logger.info("Asuming %s is new transmitter." % (str(reading)))
        new_index = Kalman.LOB + self._filter.number_of_transmitters() + \
            len(new_transmitters)
        new_transmitters.append(reading)
        self._paired_transmitters.add(new_index)
        self._paired_strengths[new_index] = [(self._filter.position(),
                                              reading[self._STRENGTH])]
        self._cycle_data[new_index] = self._cycles

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
    logger.debug("Past lobs: Calculating distance.")
    positions = {}
    for transmitter in transmitters:
      # Grab the data we need from the filtered state.
      current_lob = self._filter.state()[transmitter]
      current_position = self._filter.position()

      # To find out where they cross, we first have to translate the lobs into
      # actual lines. Start with the current one.
      intersection = self._calculate_intersection(transmitter - Kalman.LOB,
                                                  current_lob, current_position)

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
      if transmitter - Kalman.LOB >= len(self._past_lobs):
        # Use strength, this transmitter is too new.
        strength_readings[transmitter] = reading
        continue

      past_x = self._past_lobs[transmitter - Kalman.LOB][2]
      past_y = self._past_lobs[transmitter - Kalman.LOB][3]

      past_measurement_transmitters.append(transmitter)

    distances = self._distance_from_strength(strength_readings)
    logger.debug("Distance: Using past data for transmitters: %s" % \
                 (past_measurement_transmitters))
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
        # Each point on the error ellipsoid has the same layout as the state...
        hypothetical_lob = point[i]
        hypothetical_position = (point[self._X], point[self._Y])

        # Re-calculate the transmitter location using the hypothetical lob and
        # drone position from within the error region instead.
        if (i - Kalman.LOB >= len(self._past_lobs)):
          logger.debug("Using strength: %f" % (strength))
          new_location = \
              self._single_point_strength_distance(hypothetical_position,
                                                   hypothetical_lob, strength)
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
      if euclidean((current_x, current_y), (old_pos_x, old_pos_y)) >= \
         self.MIN_DISTANCE:
        use_state = state
        use_covariance = covariance
        used += 1
      else:
        # They're just going to get closer as we go up.
        break

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

  def _prune_transmitters(self, readings):
    """ Remove transmitters that were likely added erroneously.
    Args:
      readings: The raw readings from the radio. """
    self._observed_transmitters.append(len(readings))
    if len(self._observed_transmitters) > self.MAX_INNACTIVE_CYCLES:
      self._observed_transmitters.popleft()

    # If we've consistently been getting readings for fewer transmitters that we
    # have, then it's likely that not all of them are real.
    transmitters = self._filter.transmitter_positions()
    position = self._filter.position()

    # Start by eliminating transmitters that we are out of range of, as we are
    # obviously not going to get signals from these ones.
    visible_transmitters = []
    for transmitter in transmitters:
      distance = np.sqrt((transmitter[self._X] - position[self._X]) ** 2 + \
                         (transmitter[self._Y] - position[self._Y] ** 2))
      if distance <= self.RADIO_RANGE:
        visible_transmitters.append(transmitter)

    # Check if we are seeing fewer of the visible transmitters than we would
    # expect.
    for num_transmitters in self._observed_transmitters:
      if num_transmitters >= len(visible_transmitters):
        break
    else:
      # We are seeing too few transmitters.
      logger.warning("Got readings for fewer transmitters than expected.")

      # Check for a particular transmitter which we haven't gotten readings for
      # in awhile.
      for transmitter, cycle in self._cycle_data.iteritems():
        if self._cycle - cycle >= self.MAX_INNACTIVE_CYCLES:
          # This transmitter was likely the result of a single bad reading.
          logger.warning("Transmitter %d last seen on cycle %d." % \
              (transmitter, cycle))
          self._filter.remove_transmitter(transmitter)
          del self._cycle_data[transmitter]
          break

      else:
        # In this case, we probably have a situation in which there are two
        # duplicate transmitters.
        logger.info("Removing duplicate transmitter.")

        lobs = self._filter.lobs()
        intervals = \
            self._filter.lob_confidence_intervals(self.ERROR_REGION_Z_SCORE)
        sort_indices = np.argsort(lobs)

        # Find the confidence intervals with the greatest percentage overlap.
        max_overlap = 0
        left_overlap_index = None
        right_overlap_index = None
        for i in range(0, len(sort_indices) - 1):
          index = sort_indices[i]
          next_index = sort_indices[i + 1]
          interval = (lobs[index] - intervals[index],
                      lobs[index] + intervals[index])
          next_interval = (lobs[next_index] - intervals[next_index],
                           lobs[next_index] + intervals[next_index])

          raw_overlap = interval[1] - next_interval[0]
          total = next_interval[1] - interval[0]
          overlap = raw_overlap / total

          if overlap > max_overlap:
            max_overlap = overlap
            left_overlap_index = index
            right_overlap_index = next_index

        # Out of these two intervals, remove the transmitter with the larger
        # one.
        range1 = intervals[left_overlap_index] * 2
        range2 = intervals[right_overlap_index] * 2

        if range1 > range2:
          self._filter.remove_transmitter(Kalman.LOB + left_overlap_index)
          del self._cycle_data[Kalman.LOB + left_overlap_index]
        else:
          self._filter.remove_transmitter(Kalman.LOB + right_overlap_index)
          del self._cycle_data[Kalman.LOB + right_overlap_index]

  def _condense_virtual_tranmitters(self, associated):
    """ Because our RDF system registers two possible transmitter bearings 180
    degrees apart, we initially end up with "virtual" transmitters, because we
    don't know which one is correct. This method observes how the strength
    changes as the drone changes position, and uses that to infer which
    measurement is correct.
    Args:
      associated: A dictionary of associated radio readings for this cycle. The
      data in here may be modified. """
    to_delete = []
    flip_transmitters = set()
    for transmitter_index in self._paired_transmitters:
      # If we pass over a transmitter during the course of measuring strengths, we
      # end up with an ambiguous data set. A good data set should look about like
      # a straight line. Before we do anything, then, we evaluate the usefulness
      # of our data set initially using a linear regression.
      times = []
      strengths = []
      time = 0
      reading_count = 0
      # Build the x dataset, which is time, in this case measured in cycles.
      for pair in self._paired_strengths[transmitter_index]:
        if pair != None:
          # If we missed a point, we skip that cycle.
          times.append(time)
          strengths.append(pair[self._STRENGTH])
          reading_count += 1
        time += 1

      # We need a certain amount of data for this to be useful.
      if reading_count < self.MIN_DATA_FOR_REGRESSION:
        logger.warning("Not enough data from transmitter %d for regression." % \
                       (transmitter_index))
        continue

      _, _, r, _, _ = linregress(times, strengths)
      logger.debug("Got r^2 value of %f." % (r ** 2))
      if r ** 2 < self.MIN_R_SQUARED:
        logger.debug("r^2 is too big!")
        # This data is crap, so we might as well start over collecting new data.
        self._paired_strengths[transmitter_index] = \
            [self._paired_strengths[transmitter_index][0]]
        continue

      reading = associated.get(transmitter_index)
      if not reading:
        # We don't have enough data to do this.
        continue

      # First of all, see if we've moved enough in the direction of the
      # transmitter's LOB to justify testing it.
      lob = self._filter.state()[transmitter_index]
      # Turn the LOB into a unit vector.
      lob = (np.cos(lob), np.sin(lob))

      # First, compute a vector for our position change.
      seen_at_drone_position = self._paired_strengths[transmitter_index][0][0]
      old_x_pos, old_y_pos = seen_at_drone_position
      x_pos, y_pos = self._filter.position()
      d_position = (x_pos - old_x_pos, y_pos - old_y_pos)

      # Now, find the component of this vector that is in the direction of our
      # LOB.
      distance = np.dot(d_position, lob) / np.linalg.norm(lob)
      if abs(distance) < self.MIN_PAIR_ELIMINATION_DISTANCE:
        # We're still too close.
        continue

      # Check to make sure there's a large enough difference in our strengths.
      old_strength = self._paired_strengths[transmitter_index][0][1]
      strength = reading[self._STRENGTH]
      if abs(strength - old_strength) < self.MIN_STRENGTH_DIFF:
        # There's not enough of a difference in strengths to conclude anything.
        continue

      # Now, determine which version of the LOB we are picking.
      if ((distance > 0 and strength - old_strength < 0) or \
          (distance < 0 and strength - old_strength > 0)):
        # We want the version 180 degrees away.
        self._filter.flip_transmitter(transmitter_index)
        flip_transmitters.add(transmitter_index)
      # Otherwise, we want the version we have now.

      to_delete.append(transmitter_index)
      self._paired_strengths.pop(transmitter_index)

    # Remove all the transmitters from _paired_transmitters that we should have.
    for transmitter_index in to_delete:
      self._paired_transmitters.remove(transmitter_index)

    # Now, actually flip the associated measurements that need to be flipped.
    self.__flip_lobs_if_needed(associated, flip_transmitters)

  def __flip_lobs_if_needed(self, associated, flip_transmitters):
    """ A helper function that looks at a dictionary of associated LOBs, and
    flips any ones that should be flipped.
    Args:
      associated: The dictionary of associated LOBs.
      flip_transmitters: These are new transmitters that we have just decided
      need to be flipped. """
    for transmitter in flip_transmitters:
      reading = associated.get(transmitter)
      if not reading:
        # We didn't get a reading for this transmitter on this cycle.
        continue

      logger.debug("Flipping LOB for transmitter %d." % (transmitter))
      lob, strength = reading
      lob += np.pi
      lob %= 2 * np.pi

      associated[transmitter] = (lob, strength)

    # For all the rest of the transmitters, choose the option that keeps us
    # closest to our last lob. This will be correct even if we have passed right
    # over the transmitter.
    if not len(self._past_states):
      # Nothing to do here...
      return
    last_state = self._past_states.pop()
    self._past_states.append(last_state)
    for i in range(Kalman.LOB, len(last_state)):
      if i in self._paired_transmitters:
        # We're still not sure at all which version of this one is right.
        continue
      if i in flip_transmitters:
        # This one should already be correct.
        continue
      reading = associated.get(i)
      if not reading:
        # We don't have any measurement for it this cycle.
        continue

      new_lob, strength = reading
      last_lob = last_state[i]
      opposite_lob = (new_lob + np.pi) % (2.0 * np.pi)

      first_possible_diff = abs(new_lob - last_lob)
      first_possible_diff = min(first_possible_diff,
                                2.0 * np.pi - first_possible_diff)
      second_possible_diff = abs(opposite_lob - last_lob)
      second_possible_diff = min(second_possible_diff,
                                 2.0 * np.pi - second_possible_diff)

      if second_possible_diff < first_possible_diff:
        # Flip it.
        logger.debug("Flipping LOB for transmitter %d." % (i))
        new_lob += np.pi
        new_lob %= 2 * np.pi

        associated[i] = (new_lob, strength)

  def iterate(self):
    """ Runs a single iteration of the belief manager. """
    logger.debug("Starting iteration.")

    # Check what transmitters we can see, and update drone position
    # measurements.
    readings = self._fetch_data()
    logger.debug("Got raw readings: %s" % (readings))

    # Figure out which readings correspond with which transmitters.
    existing, new = self._associate_lob_readings(readings)
    logger.debug("Got new readings: %s" % (new))
    logger.debug("Got old readings: %s" % (existing))

    # Check to see if we can make any conclusions about which version of each
    # existing LOB is correct.
    self._condense_virtual_tranmitters(existing)

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

    # Remove duplicate and anomaly transmitters.
    self._prune_transmitters(readings)

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

    self._filter.update()

    # Save new error regions.
    # TODO(danielp): Uncomment this when we actually want to use error regions.
    # For now, calculating them is kind of a waste of time.
    #self._transmitter_error_regions(self.ERROR_REGION_Z_SCORE, existing, new)
    self._cycles += 1

  def _run(self):
    """ Entry point for the process. """
    logger.info("Starting BeliefManager process.")

    # Initialize the Kalman filter.
    self._fetch_data()
    while (not self._observed_position_x or not self._observed_position_y or \
           not self._observed_velocity_x or not self._observed_position_y):
      self._fetch_data()
      logger.warning("Got blank data, delaying filter initialization.")

    self._filter = Kalman((self._observed_position_x,
                           self._observed_position_y),
                          (self._observed_velocity_x,
                           self._observed_velocity_y))
    logger.info("Got initial data.")

    while True:
      # No PLL stuff is needed here, because it will automatically be
      # synchronized to the aggregator process, since it reads its input data
      # from there.
      self.iterate()
