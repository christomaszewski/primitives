import numpy as np
import yaml

class KeyPoint(yaml.YAMLObject):

	def __init__(self, point, angle, size=1, octave=1, response=1):
		self._point = point
		self._angle = angle
		self._size = size
		self._octave = octave
		self._response = response

	def offset(self, point):
		self._point += point

	def getCVKeyPointParams(self):
		params = dict(x=self.x, y=self.y, _size=self._size,
					  _angle=self._angle, _response=self._response, _octave=self._octave)

		return params

	@classmethod
	def from_cv_keypoint(cls, cvKeyPoint):
		return cls(np.asarray([cvKeyPoint.pt]), cvKeyPoint.angle, cvKeyPoint.size,
					cvKeyPoint.octave, cvKeyPoint.response)

	@property
	def point(self):
		return self._point

	@property
	def x(self):
		return self._point[0,0]

	@property
	def y(self):
		return self._point[0,1]