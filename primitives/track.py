from scipy.interpolate import interp1d
import numpy as np
import yaml
import re

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
	new_id = 0

	def __init__(self, positions=None, times=None, idNum=None):
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

		if (idNum is None):
			self._id = Track.new_id
			Track.new_id += 1
		else:
			self._id = idNum

	@classmethod
	def from_point(cls, point, time):
		numpyPoint = np.asarray(point)
		return cls([numpyPoint], [time])

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			track = yaml.load(f)

		return track

	@classmethod
	def from_file_list(cls, files):
		for filename in files:
			with open(filename, mode='r') as f:
				track = yaml.load(f)
				yield track

	@classmethod
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node, deep=True)
		positions = [np.asarray(p) for p in dict_rep['positions']]
		times = dict_rep['times']
		idNum = dict_rep['id']
		track = cls(positions, times, idNum)
		state = TrackState[dict_rep['state']]
		return track

	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes track  to yaml for output to file
			
		"""
		positions = np.asarray(data._positions).tolist()
		dict_rep = {'positions':positions, 'times':data._times, 'state':data._state.name, 'id':data._id}

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
		if (index >= len(self._positions) or len(self._positions) < 1):
			# Index out of bounds
			return None

		return (self._times[index], self._positions[index])

	def atTime(self, time):
		epsilon = 0.000001 #seconds
		if (len(self._times) < 1):
			print("Error: Empty track without observations")
			return None
		if (time < self._times[0] or self._times[-1] < time):
			print(f"time out of valid track range: {time}")
			return None

		times = np.asarray(self._times).T
		positions = np.asarray(self._positions).T

		f = interp1d(times, positions)

		return f(time)

	def addObservation(self, position, time=None):
		if (time is None):
			time = time.time()

		position = np.asarray(position)

		if (position.ndim < 1):
			print("error, inserting scalar:", position)

		self._positions.append(position)
		self._times.append(time)

		self._state = TrackState.ACTIVE

	def addObservations(self, position, time):
		self._positions.extend(position)
		self._times.extend(time)

		self._state = TrackState.ACTIVE

	def getFirstObservation(self):
		if (len(self._positions) < 1):
			return None

		return (self._times[0], self._positions[0])

	def getLastObservation(self):
		if (len(self._positions) < 1):
			return None

		return (self._times[-1], self._positions[-1])

	def size(self):
		return len(self._positions)

	def age(self):
		return self._time()

	def distance(self):
		if (len(self._positions) < 2):
			return 0.0

		startPoints = np.asarray(self._positions[:-1])
		endPoints = np.asarray(self._positions[1:])

		differences = endPoints - startPoints

		return np.sum([np.linalg.norm(diff) for diff in differences])

	def displacement(self):
		return self._displacement()

	def measureVelocity(self, minDist=0.01, method='midpoint', scoring='time'):
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
				diff = point - prevPoint

				if (np.linalg.norm(diff) < minDist):
					# points not far enough apart on track
					continue

				xVel = (point[0] - prevPoint[0]) / deltaT
				yVel = (point[1] - prevPoint[1]) / deltaT
				vel = tuple(diff/deltaT)

				measurementPoint = methodFunc(prevPoint, point)
				m = Measurement(measurementPoint, vel, score, self._id)
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

	def _prior(self, flowPrior=np.array((0.,-1.))):
		if (len(self._positions) < 2):
			return 0.0

		vec = self._positions[-1] - self._positions[0]
		vec /= np.linalg.norm(vec)

		return np.dot(flowPrior, vec)

	def _composite(self):
		kP = 0.2
		kA = 0.3
		kD = 0.1

		return (self._prior() - kP)* (kD*self._displacement() + kA*self._time())

	def _constant(self):
		return 999999

	@property
	def id(self):
		return self._id

	@id.setter
	def id(self, idNum):
		self._id = idNum

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
	def avgSpeedFast(self):
		return self._displacement()/self.age()

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



class KeyPointTrack(yaml.YAMLObject):
	""" Represents a single particle/point on an object tracked over some 
		period of time. Can be used to produce vector field measurements

		self._positions holds particle locations in list of numpy arrays
		self._times holds the times the particle was seen at the location

	"""

	yaml_tag = '!KeyPoint_Track'

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
		numpyPoint = np.asarray(point)
		return cls([numpyPoint], [time])

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
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node, deep=True)
		positions = [np.asarray(p) for p in dict_rep['positions']]
		times = dict_rep['times']
		track = cls(positions, times)
		state = TrackState[dict_rep['state']]
		return track

	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes track  to yaml for output to file
			
		"""
		positions = np.asarray(data._positions).tolist()
		dict_rep = {'positions':positions, 'times':data._times, 'state':data._state.name}

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

		position = np.asarray(position)

		if (position.ndim < 1):
			print("error, inserting scalar:", position)

		self._positions.append(position)
		self._times.append(time)

		self._state = TrackState.ACTIVE

	def addObservations(self, position, time):
		self._positions.extend(position)
		self._times.extend(time)

		self._state = TrackState.ACTIVE

	def getFirstObservation(self):
		if (len(self._positions) < 1):
			return None

		return (self._times[0], self._positions[0])

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

	def measureVelocity(self, minDist=0.0, method='midpoint', scoring='time'):
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
				diff = point - prevPoint

				if (np.linalg.norm(diff) < minDist):
					# points not far enough apart on track
					continue

				xVel = (point[0] - prevPoint[0]) / deltaT
				yVel = (point[1] - prevPoint[1]) / deltaT
				vel = tuple(diff/deltaT)

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
	def avgSpeedFast(self):
		return self._displacement()/self.age()

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