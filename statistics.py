""" Useful statistical functions. """


import numpy as np


""" Creates an n-dimensional error ellipsoid from a given covariance matrix and
center point.
Args:
  covariance: The covariance matrix to use.
  center: The center point of the ellipse. (Should be a numpy tuple.)
  z_score: How many standard deviations we want the ellipse to encompass.
  points: This function is designed to approximate the ellipse by calculating a
  number of intermediate points on the ellipse. This argument specifies how many
  intermediate points to calculate.
Returns:
  A list of all the points on the ellipse that were calculated. """
def error_ellipse(covariance, center, z_score, points):
  # Calculate eigenvalues and eigenvectors, which define the ellipse.
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)

  # Find the maximum and minimum values in each dimension.
  radii = z_score * np.sqrt(eigenvalues)
  maxima = center + radii
  minima = center - radii

  # Choose random points on the ellipse.
  selected_points = []
  dimensions = np.arange(0, len(center))
  for i in range(0, points):
    np.random.shuffle(dimensions)

    # All the terms in the ellipse equation must add to one.
    remaining = 1
    point = [0] * len(dimensions)
    for dimension in dimensions[:-1]:
      denominator = radii[dimension] ** 2

      # Assuming that all the remaining terms are zero, we know that this term
      # can be at absolute maximum whatever is remaining.
      maximum = np.sqrt(remaining * denominator)
      coordinate_value = 2 * maximum * np.random.random() - maximum
      point[dimension] = coordinate_value

      remaining -= coordinate_value ** 2 / float(denominator)

    last_dimension = dimensions[-1]
    # We only have one choice for the value of our last variable.
    denominator = radii[last_dimension] ** 2
    point[last_dimension] = np.sqrt(remaining * denominator)

    # Optionally, use the negative solution.
    if np.random.randint(0, 2):
      point[last_dimension] = -point[last_dimension]

    selected_points.append(point)

  return selected_points
