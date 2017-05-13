import numpy as np
import yaml

from enum import Enum

from .measurement import Measurement

class TrackState(Enum):
	EMPTY = 1
	ACTIVE = 2
	LOST = 3
	HISTORICAL = 4


class Track(yaml.YAMLObject):
	""" Represents a single particle/point on an object tracked over some 
		period of time. Can be used to produce vector field measurements

		self._positions holds particle locations in list of numpy arrays
		self._times holds the times the particle was seen at the location

	"""

	yaml_tag = '!Track'

	def __init__(self, positions=None, times=None):
		self._positions = []
		self._times = []

		if (positions is not None):
			self._state = TrackState.ACTIVE
			self._positions.extend(positions)
			if (times is None):
				self._times.extend(list(time.time())*len(positions))
			else:
				self._times.extend(times)
		else:
			self._state = TrackState.EMPTY

		self._lastKeyPoint = None

	@classmethod
	def from_point(cls, point, time):
		return cls([point], [time])

	@classmethod
	def from_key_point(cls, keyPoint, time):
		t = cls([np.squeeze(keyPoint.point)], [time])
		t._lastKeyPoint = keyPoint
		return t

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			return yaml.load(f)


	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes track  to yaml for output to file
			
		"""
		dict_rep = {'_positions':data._positions, '_times':data._times, '_state':data._state}

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node

	def save(self, filename):
		with open(filename, 'w') as f:
			yaml.dump(self, f)

	def __getitem__(self, index):
		""" Overrides [] operator to return the observation at index

			Returns observation:
			(time, position) where position is np.array([x,y])

		"""
		if (index >= len(self._positions) or len(self._positions < 1)):
			# Index out of bounds
			return None

		return (self._times[index], self._positions[index])

	def addKeyPointObservation(self, keyPoint, time=None):
		if (time is None):
			time = time.time()

		self._positions.append(np.squeeze(keyPoint.point))
		self._times.append(time)

		self._lastKeyPoint = keyPoint

	def addObservation(self, position, time=None):
		if (time is None):
			time = time.time()

		if (position.ndim < 1):
			print("error, inserting scalar:", position)

		self._positions.append(position)
		self._times.append(time)

		self._state = TrackState.ACTIVE

	def addObservations(self, position, time):
		self._positions.extend(position)
		self._times.extend(time)

		self._state = TrackState.ACTIVE

	def getLastObservation(self):
		if (len(self._positions) < 1):
			return None

		return (self._times[-1], self._positions[-1])

	def size(self):
		return len(self._positions)

	def age(self):
		return self._time()

	def length(self):
		return self._displacement()

	def measureVelocity(self, method='midpoint', scoring='time'):
		""" Returns list of measurements representing velocity of particle
			localizing the measurement using the method specified. Velocity
			is computed by comparing pairs on consecutive points.

			midpoint: localize the measurement on the midpoint of the segment 
			between two consecutive particle locations
			front: localize measurement on first point of consecutive point pairs
			end: localize measurement on second point of consecutive point pairs

			Should return empty list of measurements if 0 or 1 observations

			Todo: Don't recalculate measurements if track has not changed since last
			function call
		"""
		if (len(self._positions) < 2):
			return []

		methodFuncName = "_" + method
		methodFunc = getattr(self, methodFuncName, lambda p1, p2: p1)
		scoringFuncName = "_" + scoring
		scoringFunc = getattr(self, scoringFuncName, lambda: 0.0)

		score = scoringFunc()

		measurements = []

		prevPoint = None
		prevTime = None

		for timestamp, point in zip(self._times, self._positions):
			if prevPoint is not None:
				deltaT = timestamp - prevTime
				#print(point, prevPoint)

				xVel = (point[0] - prevPoint[0]) / deltaT
				yVel = (point[1] - prevPoint[1]) / deltaT
				vel = (xVel, yVel)

				measurementPoint = methodFunc(prevPoint, point)

				m = Measurement(measurementPoint, vel, score)
				measurements.append(m)

			prevPoint = point
			prevTime = timestamp

		return measurements

	def getMeasurements(self, method='midpoint', scoring='time'):
		""" Deprecated alias for measure velocity function

		"""
		print("Deprecated Function Called")
		return self.measureVelocity(method, scoring)

	# Method Functions
	def _first(self, p1, p2):
		return p1

	def _last(self, p1, p2):
		return p2

	def _midpoint(self, p1, p2):
		x = (p1[0] + p2[0]) / 2
		y = (p1[1] + p2[1]) / 2
		return (x, y)
	
	# Scoring Functinns
	def _time(self):
		# If track is 1 or less points long return 0
		if (len(self._times) < 2):
			return 0.0

		# Length of track in time
		return self._times[-1] - self._times[0]

	def _length(self):
		# Length of track in number of measurements
		return self.size()

	def _displacement(self):
		# If track is 1 or less points long return 0
		if (len(self._positions) < 2):
			return 0.0

		start = self._positions[0]
		end = self._positions[-1]

		diff = end - start
		#(end[0] - start[0], end[1] - start[1])

		return np.linalg.norm(diff)

	def _composite(self):
		kA = 0.9
		kD = 0.1

		return kA*self._time() + kD*self._displacement()

	def _constant(self):
		return 999999

	@property
	def endPoint(self):
		if (len(self._positions) < 1):
			return None

		if (self._positions[-1].ndim < 1):
			print("point dim wrong", self._positions[-1], self._positions[-1].shape)

		return self._positions[-1]

	@property
	def lastSeen(self):
		if (len(self._times) < 1):
			return None

		return self._times[-1]

	@property
	def lastKeyPoint(self):
		return self._lastKeyPoint

	@property
	def score(self):
		# Negative to use min heap as max heap
		return -self._composite()

	@property
	def positions(self):
		return self._positions

	@positions.setter
	def positions(self, positions):
		self._positions = positions

	@property
	def times(self):
		return self._times

	@property
	def state(self):
		return self._state

	@property
	def avgSpeed(self):
		# Takes long to compute !!!
		prevPoint = None
		prevTime = None

		speeds = []

		for (point, time) in zip(self._positions, self._times):
			if prevPoint is not None and prevTime is not None:
				diff = point - prevPoint
				diff = (point[0] - prevPoint[0], point[1] - prevPoint[1])
				dist = np.linalg.norm(diff)

				speeds.append(dist/(time - prevTime))

			prevPoint = point
			prevTime = time

		return np.mean(speeds)
		
		#return self._displacement()/self.age()

	@state.setter
	def state(self, newState):
		self._state = newState

	def __sub__(self, other):
		# Only subtracts matching times
		differences = []
		for t1, pt1, t2, pt2 in zip(self._times, self._positions, other.times, other.positions):
			if (t1 == t2):
				differences.append((pt1[0]-pt2[0], pt1[1]-pt2[1]))

		return differences

	def __rsub__(self, other):
		differences = []
		for t1, pt1, t2, pt2 in zip(other.times, other.positions, self._times, self._positions):
			if (t1 == t2):
				differences.append((pt1[0]-pt2[0], pt1[1]-pt2[1]))

		return differences

	def __lt__(self, other):
		return self.score < other.score

	def __gt__(self, other):
		return self.score > other.score