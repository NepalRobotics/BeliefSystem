#!/usr/bin/python


import os
import sys
import unittest


""" Runs all the unit tests.
Returns: True or False depending on whether tests succeed. """
def run_tests():
  loader = unittest.loader.TestLoader()
  suite = loader.discover("tests", top_level_dir=os.getcwd())

  test_result = unittest.TextTestRunner(verbosity=2).run(suite)
  if not test_result.wasSuccessful():
    print "ERROR: Unit tests failed."
    return False

  return True


def main():
  if not run_tests():
    sys.exit(1)


if __name__ == "__main__":
  main()
