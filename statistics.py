""" Useful statistical functions. """


from operator import add

import numpy as np


""" Creates an n-dimensional error ellipsoid from a given covariance matrix and
center point.
Args:
  covariance: The covariance matrix to use.
  center: The center point of the ellipse. (Should be a numpy array.)
  z_score: How many standard deviations we want the ellipse to encompass.
  points: This function is designed to approximate the ellipse by calculating a
  number of intermediate points on the ellipse. This argument specifies how many
  intermediate points to calculate.
Returns:
  A list of all the points on the ellipse that were calculated. """
def error_ellipse(covariance, center, z_score, points):
  # Calculate eigenvalues and eigenvectors, which define the ellipse.
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)
  # Sometimes due to floating-point innacuracies, or eigenvalues get slightly
  # negative. If this happens, bump them back to a bit above zero. (They can't
  # be zero exactly, because they end up in the denominator.)
  for value in np.nditer(eigenvalues, op_flags=["readwrite"]):
    if value < 0:
      if 0 - value < 0.001:
        value[...] = value * -1
      else:
        raise ValueError("Negative eigenvalue of covariance matrix!")

  # Find the semi-axes in each direction.
  print "Eigvals: %s" % (eigenvalues)
  radii = z_score * np.sqrt(eigenvalues)

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

    # Shift so it's centered where we want it.
    point = map(add, point, center)

    selected_points.append(point)

  return selected_points
