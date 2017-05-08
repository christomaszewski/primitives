import yaml

class Measurement(yaml.YAMLObject):
	""" Represents a single measurement of a field at a point

	"""

	yaml_tag = '!Measurement'

	def __init__(self, point, value, score=0.0):
		self._point = point
		self._value = value
		self._score = score

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			return yaml.load(f)

	@classmethod
	def to_yaml(cls, dumper, data):

		pointNode = data._point.tolist()
		valueNode = data._value.tolist()

		dict_rep = {'_point':pointNode, '_value':valueNode, '_score':data._score}

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node

	@property
	def point(self):
		return self._point

	@property
	def value(self):
		return self._value
	
	@property
	def score(self):
		return self._score

	def __cmp__(self, other):
		# Negative so we can use minheap as maxheap
		return -(cmp(self.score, other.score))

	def __lt__(self, other):
		return self.score < other.score

	def __gt__(self, other):
		return self.score > other.score

	def __radd__(self, other):
		return self.score + other

	def __add__(self, other):
		return self.score + other

	def __str__(self):
		return "[" + str(self.point) + "," + str(self.value) + "]"

	def __repr__(self):
		return str(self)