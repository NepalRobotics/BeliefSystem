""" Tests for the aggregator process. """


import unittest

from Core.message_object import VehicleState, RadioState, BeliefMessage
from Core.utils import TestQueue

from belief_manager import BeliefManager
import aggregator


class AggregatorTests(unittest.TestCase):
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
    self.assertEqual(drone_data.latitude, message.latitude)
    self.assertEqual(drone_data.longitude, message.longitude)
    self.assertEqual(drone_data.velocity_array[0], message.x_velocity)
    self.assertEqual(drone_data.velocity_array[1], message.y_velocity)

  def test_oversample(self):
    """ Tests oversampling of autopilot data. """
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

    # Check the output.
    message = self.__belief_queue.get()
    self.assertEqual(3, message.latitude)
    self.assertEqual(3, message.longitude)
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
    self.assertEqual(None, message.latitude)
    self.assertEqual(None, message.longitude)
    self.assertEqual(None, message.x_velocity)
    self.assertEqual(None, message.y_velocity)

    self.__aggregator.iterate()

    # Next time, though, we should get something.
    message = self.__belief_queue.get()
    self.assertEqual((radio_data.lob, radio_data.strength), message.radio_data[0])
    self.assertEqual(drone_data.latitude, message.latitude)
    self.assertEqual(drone_data.longitude, message.longitude)
    self.assertEqual(drone_data.velocity_array[0], message.x_velocity)
    self.assertEqual(drone_data.velocity_array[1], message.y_velocity)
