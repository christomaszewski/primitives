import yaml

class Measurement(yaml.YAMLObject):
	""" Represents a single measurement of a field at a point

	"""

	yaml_tag = '!Measurement'
	new_id = 0

	def __init__(self, point, value, score=0.0, idNum=None):
		self._point = point
		self._value = value
		self._score = score

		if (idNum is None):
			self._id = Measurement.new_id
			Measurement.new_id += 1
		else:
			self._id = idNum

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

	def save(self, filename):
		with open(filename, 'w') as f:
			yaml.dump(self, f)

	@property
	def point(self):
		return self._point

	@property
	def value(self):
		return self._value
	
	@property
	def score(self):
		return -self._score

	@property
	def id(self):
		return self._id

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