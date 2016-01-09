""" Tests for the aggregator process. """


import time

from Utils.message_object import VehicleState, RadioState, BeliefMessage
from Utils.utils import TestQueue

from belief_manager import BeliefManager
import aggregator
import tests


class AggregatorTests(tests.BaseTest):
  """ Tests for the Aggregator class. """

  def setUp(self):
    self.__mav_queue = TestQueue()
    self.__radio_queue = TestQueue()
    self.__belief_queue = TestQueue()

    self.__aggregator = aggregator.Aggregator(self.__mav_queue,
                                              self.__radio_queue,
                                              self.__belief_queue)

  def test_iterate(self):
    """ Tests that we can run one iteration. """
    # Put some stuff on the queue for it to get.
    drone_data = VehicleState("test")
    drone_data.latitude = 1
    drone_data.longitude = 1
    drone_data.velocity_array = [1, 1]
    self.__mav_queue.put(drone_data)

    radio_data = RadioState(0, 1.0)
    self.__radio_queue.put(radio_data)

    # Run one iteration.
    self.__aggregator.iterate()

    # See what it wrote to the belief box.
    message = self.__belief_queue.get()
    self.assertEqual([(radio_data.lob, radio_data.strength)],
                     message.radio_data)
    # Since we ran one iteration, we should be at the origin.
    self.assertEqual(0, message.x_pos)
    self.assertEqual(0, message.y_pos)
    self.assertEqual(drone_data.velocity_array[0], message.x_velocity)
    self.assertEqual(drone_data.velocity_array[1], message.y_velocity)

  def test_oversample(self):
    """ Tests oversampling of autopilot data. """
    # Set the initial reference point for positions.
    start_data = VehicleState("test")
    start_data.latitude = 3
    start_data.longitude = 3
    start_data.velocity_array = [3, 3]
    self.__mav_queue.put(start_data)
    self.__aggregator.iterate()
    self.__belief_queue.get()
    # Wait for a cycle so are messages don't have a "past" timestamp and aren't
    # ignored.
    time.sleep(BeliefManager.CYCLE_LENGTH)

    # Make some test data.
    drone_data1 = VehicleState("test")
    drone_data2 = VehicleState("test")

    drone_data1.latitude = 2
    drone_data1.longitude = 2
    drone_data1.velocity_array = [2, 2]

    drone_data2.latitude = 4
    drone_data2.longitude = 4
    drone_data2.velocity_array = [4, 4]

    self.__mav_queue.put(drone_data1)
    self.__mav_queue.put(drone_data2)

    # Run one iteration.
    self.__aggregator.iterate()

    # Check the output. Since we set the origin to what we think things should
    # be, they should come out zeroed.
    message = self.__belief_queue.get()
    self.assertEqual(0, message.x_pos)
    self.assertEqual(0, message.y_pos)
    self.assertEqual(3, message.x_velocity)
    self.assertEqual(3, message.y_velocity)

  def test_radio_aggregation(self):
    """ Tests that it buffers and outputs all radio data. """
    # Make some test data.
    radio_data1 = RadioState(0, 1.0)
    radio_data2 = RadioState(0.5, 0.5)
    self.__radio_queue.put(radio_data1)
    self.__radio_queue.put(radio_data2)

    # Run one iteration.
    self.__aggregator.iterate()

    # Check the output.
    message = self.__belief_queue.get()
    expected = [(radio_data1.lob, radio_data1.strength),
                (radio_data2.lob, radio_data2.strength)]
    self.assertEqual(expected, message.radio_data)

  def test_future_buffering(self):
    """ Tests that it can save data and give it to us on the cycle it should.
    """
    radio_data = RadioState(0, 1.0)
    drone_data = VehicleState("test")
    drone_data.latitude = 1
    drone_data.longitude = 1
    drone_data.velocity_array = [1, 1]

    # Put it in the future.
    radio_data.time_created += BeliefManager.CYCLE_LENGTH * 1.5
    drone_data.time_created += BeliefManager.CYCLE_LENGTH * 1.5
    self.__radio_queue.put(radio_data)
    self.__mav_queue.put(drone_data)

    # Run one iteration.
    self.__aggregator.iterate()

    # We should have an empty message this time.
    message = self.__belief_queue.get()
    self.assertEqual([], message.radio_data)
    self.assertEqual(None, message.x_pos)
    self.assertEqual(None, message.y_pos)
    self.assertEqual(None, message.x_velocity)
    self.assertEqual(None, message.y_velocity)

    self.__aggregator.iterate()

    # Next time, though, we should get something.
    message = self.__belief_queue.get()
    self.assertEqual((radio_data.lob, radio_data.strength), message.radio_data[0])
    # Once again, position should be zeroed because we should be at the origin.
    self.assertEqual(0, message.x_pos)
    self.assertEqual(0, message.y_pos)
    self.assertEqual(drone_data.velocity_array[0], message.x_velocity)
    self.assertEqual(drone_data.velocity_array[1], message.y_velocity)

  def test_relative_positioning(self):
    """ Tests that converting WGS84 coordinates to a relative position in meters
    works properly. """
    # Set the initial reference point for positions.
    start_data = VehicleState("test")
    start_data.latitude = 0
    start_data.longitude = 0
    start_data.velocity_array = [1, 1]
    self.__mav_queue.put(start_data)
    self.__aggregator.iterate()
    self.__belief_queue.get()
    # Wait for one cycle so our next packet doesn't appear to be coming from the
    # "past".
    time.sleep(BeliefManager.CYCLE_LENGTH)

    # Now, set a new position.
    drone_data = VehicleState("test")
    drone_data.latitude = 1
    drone_data.longitude = 0
    drone_data.velocity_array = [1, 1]
    self.__mav_queue.put(drone_data)
    self.__aggregator.iterate()

    # See where it thinks we are.
    message = self.__belief_queue.get()
    self.assertEqual(0, message.x_pos)
    # The expected number here comes from a web tool for calculating distances
    # between points on the earth's surface.
    self._assert_near(111230, message.y_pos, 111230 * 0.003)
