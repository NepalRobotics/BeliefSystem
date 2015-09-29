""" Contains basic functionality used by multiple unit tests. """


import unittest


class BaseTest(unittest.TestCase):
  """ A superclass for all test cases that defines some useful methods. """
  def _assert_near(self, expected, actual, error):
    """ Makes sure that a paremeter is within a cetain amount of something else.
    Args:
      expected: The value we expected.
      actual: The value we got.
      error: The maximum acceptable deviation between expected and actual. """
    self.assertLess(abs(expected - actual), error)
