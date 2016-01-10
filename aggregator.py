import logging
import Queue
import time

import numpy as np

from Utils.message_object import BeliefMessage
from Utils.process import Process
from Utils.utils import PhasedLoopLimiter

from belief_manager import BeliefManager


logger = logging.getLogger(__name__)


def _moving_average(sample, count, average):
  """ A simple and efficient moving average formula, taken from here:
  http://www.daycounter.com/LabBook/Moving-Average.phtml
  Args:
    sample: The sample we are incorporating.
    count: The number of all total samples, including this one.
    average: The current average, not including this sample.
  Returns:
    The new moving average. """
  total = average * count
  new_total = total + sample - average
  return new_total / count

def _make_or_add(dictionary, key, item):
  """ If the key already exists, it appends item to the value. Otherwise, it
  creates a new list for key and adds item.
  Args:
    dictionary: The dictionary to add stuff to.
    key: The key to use or add.
    item: The item to add. """
  if not dictionary.get(key):
    dictionary[key] = [item]
    return

  dictionary[key].append(item)


class Aggregator(Process):
  """ Aggregates data from both the radio and the autopilot, and chooses the
  best to send along to the BeliefManager. """

  # The maximum number of messages it can read from any queue in one iteration.
  MAX_MESSAGE_READ = 1000
  # The radius of the earth. (km)
  EARTH_RADIUS = 6371

  def __init__(self, mav_queue, radio_queue, belief_queue):
    """ Args:
      mav_queue: The queue for receiving autopilot data.
      radio_queue: The queue for receiving radio data.
      belief_queue: The queue for sending data to the BeliefManager process. """
    super(Aggregator, self).__init__()

    self.__mav_queue = mav_queue
    self.__radio_queue = radio_queue
    self.__belief_queue = belief_queue

    self.__limitter = PhasedLoopLimiter(BeliefManager.CYCLE_LENGTH)

    # A somewhat arbitrary time that we define as denoting cycle zero.
    self.__cycle_zero = time.time()
    self.__cycle = 1
    # These are dictionaries that store all data based on which cycle it comes
    # from.
    self.__future_radio = {}
    self.__future_mav = {}

    # Our initial latitude and longitude. It doesn't actually matter where these
    # are really, but we set them to our initial position to keep the numbers
    # small.
    self.__start_latitude = None
    self.__start_longitude = None

  def __cycle_from_time(self, time):
    """ Gets the cycle number we are on based on where cycle_zero denotes the
    start of the first cycle.
    Args:
      time: The time we are converting to a cycle.
    Returns:
      The cycle corresponding to this time. """
    cycle = (time - self.__cycle_zero) / BeliefManager.CYCLE_LENGTH
    if cycle - int(cycle) > 0.0001:
      # If it doesn't go in perfectly, we should add one, because it's
      # technically in the next cycle.
      cycle += 1

    return max(int(cycle), 1)

  def __wgs_to_offset(self, latitude, longitude):
    """ Converts a GPS reading from the WGS84 format to an offset in meters
    from a defined starting location.
    It uses the formula found here:
    http://andrew.hedges.name/experiments/haversine/
    Args:
      latitude: The latitude of the point to covert.
      longitude: The longitude of the point to convert.
    Returns:
      A tuple containing the converted point in meters, with the first item
      representing the x (longitude) direction, and the second representing the
      y (latitude) direction. """
    # Convert everything to radians to make it work right.
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)

    delta_lat = latitude - self.__start_latitude
    delta_lon = longitude - self.__start_longitude

    a_y = np.sin(delta_lat / 2.0) ** 2
    c_y = 2.0 * np.arctan2(np.sqrt(a_y), np.sqrt(1 - a_y))
    a_x = np.cos(self.__start_latitude) * np.cos(latitude) * \
          np.sin(delta_lon / 2.0) ** 2
    c_x = 2.0 * np.arctan2(np.sqrt(a_x), np.sqrt(1 - a_x))

    # Get distance in each component.
    x = self.EARTH_RADIUS * 1000 * c_x
    y = self.EARTH_RADIUS * 1000 * c_y

    return (x, y)

  def iterate(self):
    """ Runs a single iteration of the process. This should be synched with the
    frequency at which BeliefManager is run. """
    logger.debug("Running iteration %d." % (self.__cycle))

    # Get and buffer radio data.
    for i in range(0, self.MAX_MESSAGE_READ):
      if self.__radio_queue.empty():
        break

      radio_message = self.__radio_queue.get_nowait()

      # Check if message is outdated.
      message_cycle = self.__cycle_from_time(radio_message.time_created)
      if message_cycle < self.__cycle:
        # This really shouldn't happen at all, unless, for some reason, our
        # queue has rediculous latency.
        logger.warning("Dropping radio packet from previous cycle.")
        continue

      # Classify the message based on its timestamp.
      if message_cycle > self.__cycle:
        # This belongs in the future, so we'll save it for later.
        logger.debug("Saving radio packet for cycle %d." % (message_cycle))
        _make_or_add(self.__future_radio, message_cycle, radio_message)
        continue

      # It belongs in the current cycle, add it there.
      _make_or_add(self.__future_radio, self.__cycle, radio_message)
    else:
      # We read too many messages.
      logger.error("Exceeded message read limit on radio queue!")

    # Put all data for this cycle into the proper format.
    radio_buffer = []
    cycle_data = self.__future_radio.pop(self.__cycle, [])
    for saved_data in cycle_data:
      radio_tuple = (saved_data.lob, saved_data.strength)
      radio_buffer.append(radio_tuple)

    if not radio_buffer:
      logger.warning("Got no radio data for this cycle.")

    # Get and process drone position data.
    for i in range(0, self.MAX_MESSAGE_READ):
      if self.__mav_queue.empty():
        break

      mav_message = self.__mav_queue.get_nowait()

      # Check if message is outdated.
      message_cycle = self.__cycle_from_time(mav_message.time_created)
      if message_cycle < self.__cycle:
        # This really shouldn't happen at all, unless, for some reason, our
        # queue has rediculous latency.
        logger.warning("Dropping MAV packet from previous cycle.")
        continue

      # Classify the message based on its timestamp.
      if message_cycle > self.__cycle:
        # This belongs in the future, so we'll save it for later.
        logger.debug("Saving MAV packet for cycle %d." % (message_cycle))
        _make_or_add(self.__future_mav, message_cycle, mav_message)
        continue

      # It belongs in the current cycle, add it there.
      _make_or_add(self.__future_mav, self.__cycle, mav_message)
    else:
      # We read too many messages.
      logger.error("Exceeded message read limit on MAV queue!")

    # Build outgoing message.
    message = BeliefMessage()

    # Format autopilot data for this cycle correctly.
    cycle_data = self.__future_mav.pop(self.__cycle, [])
    if cycle_data:
      # If we have no data, we want to keep everything None.
      message.zero()
    message.radio_data = radio_buffer

    count = 1
    average_latitude = 0
    average_longitude = 0
    average_x_vel = 0
    average_y_vel = 0
    for saved_data in cycle_data:
      # Average vehicle state data. (We might as well oversample to minimize
      # variation...)
      average_latitude = _moving_average(saved_data.latitude, count,
                                         average_latitude)
      average_longitude = _moving_average(saved_data.longitude, count,
                                          average_longitude)
      average_x_vel = _moving_average(saved_data.velocity_array[0],
                                      count, average_x_vel)
      average_y_vel = _moving_average(saved_data.velocity_array[1],
                                      count, average_y_vel)
      count += 1

    # Convert data from WGS84 to an offset in meters.
    if cycle_data:
      if (self.__start_latitude == None or self.__start_longitude == None):
        # Set the initial position.
        self.__start_latitude = np.radians(average_latitude)
        self.__start_longitude = np.radians(average_longitude)
        logger.debug("Setting initial location: (%f, %f)" % \
                    (self.__start_longitude, self.__start_latitude))

      message.x_pos, message.y_pos = self.__wgs_to_offset(average_latitude,
                                                          average_longitude)
      message.x_velocity = average_x_vel
      message.y_velocity = average_y_vel

    if count == 1:
      logger.warning("Got no drone position data for this cycle.")

    try:
      self.__belief_queue.put_nowait(message)
    except Queue.Full:
      logger.error("Belief queue is full!")

    self.__cycle += 1

  def _run(self):
    logger.info("Started aggregator process.")

    # Set the current cycle we are on. In the future, it may not track exactly
    # with the "outside" times, but it is what the BeliefManager sees.
    self.__cycle = self.__cycle_from_time(time.time())

    while True:
      self.__limitter.limit()
      self.iterate()
