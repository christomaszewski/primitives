import yaml
import json
import os

class MeasurementJSONEncoder(json.JSONEncoder):

	def default(self, data):
		dict_rep = {'point':data.point, 'value':data.value, 'score':data.score, 'idNum':data.id, 'time':data.time}

		return dict_rep

	@classmethod
	def decode(cls, json_dict):
		return Measurement(**json_dict)


class Measurement(yaml.YAMLObject):
	""" Represents a single measurement of a field at a point

	"""

	yaml_tag = '!Measurement'
	new_id = 0
	json_encoder = MeasurementJSONEncoder

	def __init__(self, point, value, score=0.0, idNum=None, time=None):
		self._point = point
		self._value = value
		self._score = score
		self._time = time

		if (idNum is None):
			self._id = Measurement.new_id
			Measurement.new_id += 1
		else:
			self._id = idNum

	@classmethod
	def from_file(cls, filename):
		extension = os.path.splitext(filename)[1][1:]
		with open(filename, mode='r') as f:
			if extension == 'yaml':
				m = yaml.load(f)
			elif extension == 'json':
				m = json.load(f, object_hook=MeasurementJSONEncoder.decode)
			else:
				print(f"Error: Unrecognized extension {extension}, supported extensions are yaml and json")
				m = None

		return m

	def save(self, filename):
		extension = os.path.splitext(filename)[1][1:]
		with open(filename, 'w') as f:
			if extension == 'yaml':
				yaml.dump(self, f)
			elif extension == 'json':
				json.dump(self, f, cls=Measurement.json_encoder, indent=2)
			else:
				print(f"Error: Unrecognized extension {extension}, supported extensions are yaml and json")

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

	@property
	def time(self):
		return self._time

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