""" Tests for the statistics functions. """


import unittest

import numpy as np

import statistics
import tests


class ErrorEllipseTests(tests.BaseTest):
  """ Tests for the error ellipse function. """
  def test_2d(self):
    """ Tests that it can compute points on a basic 2D error ellipse. """
    test_covariance = np.array([[1, 0], [0, 2]])
    test_center = (0, 0)
    test_z = 1.96
    test_points = 1000

    # These are the values we'd expect for the semi-axes. Rotation is trivial,
    # since all our covariances are zero.
    eigenvalues, _ = np.linalg.eigh(test_covariance)
    radii = test_z * np.sqrt(eigenvalues)

    points = statistics.error_ellipse(test_covariance, test_center, test_z,
                                      test_points)
    for point in points:
      # Check that these points actually work in the ellipse equation.
      terms = np.array(point) ** 2 / radii ** 2
      result = np.sum(terms)

      self._assert_near(1, result, 0.001)
