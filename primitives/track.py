import scipy as sp
from scipy.interpolate import interp1d
import numpy as np
import yaml
import json
import os

from enum import Enum

from .measurement import Measurement

class TrackState(Enum):
	EMPTY = 1
	ACTIVE = 2
	LOST = 3
	HISTORICAL = 4

class TrackJSONEncoder(json.JSONEncoder):

	def default(self, data):
		positions = np.asarray(data.positions).tolist()
		dict_rep = {'positions':positions, 'times':data.times, 'state':data.state.name, 'id':data.id}

		return dict_rep

	@classmethod
	def decode(cls, json_dict):
		positions = [np.asarray(p) for p in json_dict['positions']]
		times = json_dict['times']
		idNum = json_dict['id']
		track = Track(positions, times, idNum)
		track.state = TrackState[json_dict['state']]
		return track


class Track(yaml.YAMLObject):
	""" Represents a single particle/point on an object tracked over some 
		period of time. Can be used to produce vector field measurements

		self._positions holds particle locations in list of numpy arrays
		self._times holds the times the particle was seen at the location

	"""

	yaml_tag = '!Track'
	new_id = 0
	json_encoder = TrackJSONEncoder

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
		extension = os.path.splitext(filename)[1][1:]
		with open(filename, mode='r') as f:
			if extension == 'yaml':
				track = yaml.load(f)
			elif extension == 'json':
				track = json.load(f, object_hook=TrackJSONEncoder.decode)
			else:
				print(f"Error: Unrecognized extension {extension}, supported extensions are yaml and json")
				track = None

		return track

	@classmethod
	def from_file_list(cls, files):
		for filename in files:
			yield cls.from_file(filename)

	@classmethod
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node, deep=True)
		positions = [np.asarray(p) for p in dict_rep['positions']]
		times = dict_rep['times']
		idNum = dict_rep['id']
		track = cls(positions, times, idNum)
		track.state = TrackState[dict_rep['state']]
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
		extension = os.path.splitext(filename)[1][1:]
		with open(filename, 'w') as f:
			if extension == 'yaml':
				yaml.dump(self, f)
			elif extension == 'json':
				json.dump(self, f, cls=Track.json_encoder, indent=2)
			else:
				print(f"Error: Unrecognized extension {extension}, supported extensions are yaml and json")

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

	def measureVelocityWithMass(self, mass, bx=5.5, by=0.0275, minDist=5., scoring='time'):
		""" Returns list of measurements representing velocity of particle
			localizing the measurement using the center point of the computation. Velocity
			is computed by comparing pairs on consecutive points and using drag formula.

			Should return empty list of measurements if 0 or 1 observations

			Todo: Don't recalculate measurements if track has not changed since last
			function call

			Todo: Consider adding mass parameter to particle tracks
		"""

		if (len(self._positions) < 3):
			return []

		scoringFuncName = "_" + scoring
		scoringFunc = getattr(self, scoringFuncName, lambda: 0.0)

		score = scoringFunc()

		measurements = []

		prevPoint = None
		prevTime = None
		midPoint = None
		midTime = None

		for timestamp, point in zip(self._times, self._positions):
			if prevPoint is not None and midPoint is not None:
				deltaT1 = midTime - prevTime
				deltaT2 = timestamp - midTime
				dx1 = midPoint - prevPoint
				dx2 = point - midPoint
				v1 = dx1 / deltaT1
				v2 = dx2 / deltaT2
				a = (v1 - v2)/(deltaT1 + deltaT2)
				#print(point, prevPoint)

				if np.linalg.norm(dx1) < minDist:
					midPoint = point
					midTime = timestamp
					continue
				elif np.linalg.norm(dx2) < minDist:
					# points not far enough apart on track
					continue

				ux = a[0] * mass / bx
				uy = a[1] * mass / by

				vel = (ux, uy)
				m = Measurement(midPoint, vel, score, self._id)
				measurements.append(m)

			prevPoint = midPoint
			prevTime = midTime
			midPoint = point
			midTime = timestamp

		return measurements

	def smooth(self, *, method="savitzkyGolay", **methodArgs):
		""" Externally accessible function to smooth track using specified method with
			arguments specified in methodArgs

			savitzkyGolay: Savitzky-Golay Filtering
			movingAverage: Sliding window moving average

			returns points in smoothed trajectory
			
			Todo: add spline smoothing, add option to save smoothed values as track values
		"""
		methodFuncName = "_" + method
		methodFunc = getattr(self, methodFuncName, self._savitzkyGolay)

		return methodFunc(**methodArgs)

	def _movingAverage(self, windowSize=45):
		positions = np.asarray(self._positions)
		x = positions[:, 0]
		y = positions[:, 1]

		cumsumX = np.cumsum(np.insert(x, 0, 0)) 
		avgX = (cumsumX[windowSize:] - cumsumX[:-windowSize]) / windowSize

		cumsumY = np.cumsum(np.insert(y, 0, 0)) 
		avgY = (cumsumY[windowSize:] - cumsumY[:-windowSize]) / windowSize

		points = [np.asarray((xi, yi)) for xi,yi in zip(avgX,avgY)]

		return points

	def _savitzkyGolay(self, *, windowSize=45, order=3, deriv=0):
		# Savitzky-Golay Filtering to smooth trajectory
		from math import factorial

		order_range = range(order+1)
		half_window = (windowSize - 1) // 2

		deriv = 0
		rate = 1

		b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
		m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)


		epsilon = 0.000001 #seconds
		if (len(self._times) < 1):
			print("Error: Empty track without observations")
			return None

		times = np.asarray(self._times).T
		positions = np.asarray(self._positions)
		x = positions[:, 0].T
		y = positions[:, 1].T

		firstvalsY = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
		lastvalsY = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
		y = np.concatenate((firstvalsY, y, lastvalsY))
		y = np.convolve( m[::-1], y, mode='valid')


		firstvalsX = x[0] - np.abs(x[1:half_window+1][::-1] - x[0])
		lastvalsX = x[-1] + np.abs(x[-half_window-1:-1][::-1] - x[-1])
		x = np.concatenate((firstvalsX, x, lastvalsX))
		x = np.convolve( m[::-1], x, mode='valid')

		points = [np.asarray((xi, yi)) for xi,yi in zip(x,y)]

		return points

	def _movingAveragePreFilter(self, scoringFunc, **methodArgs):
		points = self._movingAverage(**methodArgs)

		# match up points to times such that times correspond to middle of window for computed point
		lengthMismatch = len(self._positions) - len(points) + 1
		offset = lengthMismatch // 2

		times = np.asarray(self._times)

		# if mismatch is odd
		if lengthMismatch & 0x1:
			times = times[offset:-offset]
		else:
			# mismatch is even, average two center times
			times = (times[offset-1:-offset] + times[offset:-offset+1])/2.

		return self._finiteDifference(scoringFunc, points, times)

	def _savitzkyGolayPreFilter(self, scoringFunc, **methodArgs):
		points = self._savitzkyGolay(**methodArgs)

		return self._finiteDifference(scoringFunc, points, self._times)

	def _computeDenoisedDerivative(self, data, alpha, dt, maxIter=10000, eps=0.5):
		"""
		Adapted from Total Variation Regularization algorithm developed by:
		Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data,"
		ISRN Applied Mathematics, Vol. 2011, Article ID 164564, 2011. 
		"""

		n = len(data)

		# Construct differentiation matrix
		c = np.ones(n + 1) / dt
		D = sp.sparse.spdiags([-c, c], [0, 1], n, n + 1)

		DT = D.transpose()

		# Define chop function to drop first element of array
		chop = lambda x: x[1:]

		# Construct antidifferentiation operator and its adjoint
		A = lambda x: chop(np.cumsum(x) - 0.5 * (x + x[0])) * dt
		AT = lambda w: (sum(w) * np.ones(n + 1) - np.transpose(np.concatenate(([sum(w) / 2.0], np.cumsum(w) - w / 2.0)))) * dt

		# Inialize derivatives to finite differences
		u = np.concatenate(([0], np.diff(data), [0]))

		# Adjust because Au(0) == 0
		ATb = AT(data[0] - data)

		for i in range(maxIter):
			# Diagonal matrix of weights, for linearizing E-L equation
			Q = sp.sparse.spdiags(1. / (np.sqrt((D * u)**2 + eps)), 0, n, n)
			# Linearized diffusion matrix, also approximation of Hessian
			L = dt * DT * Q * D

			# Gradient of functional
			g = AT(A(u)) + ATb + alpha * L * u

			# Prepare to solve linear equation
			tolerance = 1e-4
			maxSolverIter = 100

			# Simple preconditioner
			P = alpha * sp.sparse.spdiags(L.diagonal() + 1, 0, n + 1, n + 1)

			linop = lambda v: (alpha * L * v + AT(A(v)))
			linop = sp.sparse.linalg.LinearOperator((n + 1, n + 1), linop)

			[s, info_i] = sp.sparse.linalg.cg(linop, g, None, tolerance, maxSolverIter, None, P)
			u -= s

			if info_i == 0 and i > 10:
				return u

		return None


	def _totalVariationRegularization(self, scoringFunc):
		score = scoringFunc()

		measurements = []

		dt = self._times[1] - self._times[0]

		points = np.array(self._positions)
		x = np.array(points[:,0])
		y = np.array(points[:,1])

		dx = self._computeDenoisedDerivative(x, 5e-2, dt)
		dy = self._computeDenoisedDerivative(y, 5e-2, dt)

		# Trim first derivative and last 2 derivatives
		dx = dx[1:-2]
		dy = dy[1:-2]

		# Trim first and last times
		times = self._times[1:-1]

		if dx is None or dy is None:
			print("Computation Failed!")
			return None

		measurements = []
		for xV, yV, point, time in zip(dx.tolist(), dy.tolist(), self._positions[1:-1], times):
			measurements.append(Measurement(tuple(point), (xV, yV), score, self._id, time))

		return measurements

	def _finiteDifference(self, scoringFunc, points=None, times=None):
		if points is None:
			points = self._positions

		if times is None:
			times = self._times

		score = scoringFunc()

		measurements = []

		prevPoint = None
		prevTime = None
		midPoint = None
		midTime = None

		for timestamp, point in zip(times, points):
			if prevPoint is not None:
				if midPoint is None:
					midPoint = point
					midTime = timestamp
					continue

				deltaT = timestamp - prevTime
				diff = point - prevPoint

				vel = tuple(diff/deltaT)

				measurementPoint = tuple(midPoint)
				m = Measurement(measurementPoint, vel, score, self._id, midTime)
				measurements.append(m)
			

			prevPoint = midPoint
			prevTime = midTime
			midPoint = point
			midTime = timestamp

		return measurements

	def _minDistDifference(self, scoringFunc, minDist=0.1):
		score = scoringFunc()

		measurements = []

		prevPoint = None
		prevTime = None

		for timestamp, point in zip(self._times, self._positions):
			if prevPoint is not None:
				deltaT = timestamp - prevTime
				diff = point - prevPoint

				if (np.linalg.norm(diff) < minDist):
					# points not far enough apart on track
					continue
				
				vel = tuple(diff/deltaT)
				midPoint = (point + prevPoint) / 2.
				midTime = prevTime + deltaT

				measurementPoint = tuple(midPoint)
				m = Measurement(measurementPoint, vel, score, self._id, midTime)
				measurements.append(m)
			

			prevPoint = point
			prevTime = timestamp

		return measurements

	def measureVelocity(self, *, method='finiteDifference', scoring='time', **methodArgs):
		""" Returns list of measurements representing velocity of particle
			computed using the method specified

			finiteDifference: standard discrete differentiation using difference between two points
			minDistDifference: finite difference method except imposes a minimum distance between the two points used
			totalVariationRegularization: used method for differentiation of noisy data described by Chartrand
			savitzkyGolayPreFilter: Smooths trajectory using Savitzky-Golay filtering then uses finite difference method
			movingAveragePreFilter: Smooths trajectory using moving average then uses finite difference method

			methodArgs: passed as arguments to specified method

			Should return empty list of measurements if 0 or 1 observations

			Todo: Don't recalculate measurements if track has not changed since last
			function call
		"""
		if (len(self._positions) < 3):
			return []

		methodFuncName = "_" + method
		methodFunc = getattr(self, methodFuncName, self._finiteDifference)
		scoringFuncName = "_" + scoring
		scoringFunc = getattr(self, scoringFuncName, lambda: 0.0)

		return methodFunc(scoringFunc, **methodArgs)

	def priorCorrespondence(self, **kwargs):
		return self._prior(**kwargs)
	
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