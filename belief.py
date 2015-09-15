""" Contains a structure that will be used to represent a belief. """


""" Contains meta-information about a belief. """
class BeliefHeader(object):
  def __init__(self, **kwargs):
    # Timestamp for the belief in seconds from the epoch.
    self.timestamp = kwargs.get("timestamp")
    # Which drone the belief is coming from.
    self.drone = kwargs.get("drone")


""" Represents an Avalanche Transmitter that this drone is in contact with. """
class AvalancheTransmitter(object):
  def __init__(self, **kwargs):
    # A unique identifier for the transmitter.
    self.name = kwargs.get("name")
    # Our current location of bearing (LOB) on the transmitter.
    self.lob = kwargs.get("lob")
    # Where we currently think the transmitter is, in GPS coordinates. (x, y)
    self.position = kwargs.get("position")


""" Contains the actual belief information. """
class Belief(object):
  def __init__(self, transmitters=[], **kwargs):
    # The header for this belief.
    self.header = kwargs.get("header")

    # The current position of this drone, in GPS coordinates. (x, y)
    self.position = kwargs.get("position")
    # All the Avalanche Transmitters we can see.
    self.transmitters = transmitters
