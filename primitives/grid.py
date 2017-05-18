import numpy as np
import yaml

class Grid(yaml.YAMLObject):

	yaml_tag = '!Grid'

	def __init__(self, xDist, yDist, xCellCount, yCellCount=None, origin=(0,0)):
		self._xDist = xDist
		self._yDist = yDist
		self._xCellCount = xCellCount

		if (yCellCount is None):
			self._yCellCount = self._xCellCount
		else:
			self._yCellCount = yCellCount

		self._totalCells = self._xCellCount * self._yCellCount

		self._xCellWidth = self._xDist / self._xCellCount
		self._xCellHalfWidth = self._xCellWidth / 2.0

		self._yCellWidth = self._yDist / self._yCellCount
		self._yCellHalfWidth = self._yCellWidth / 2.0

		self._xOrigin, self._yOrigin = origin


	@classmethod
	def from_bounds(cls, bounds, cellSize=(5,5)):
		""" Builds a grid object from a list of bounds and an approximate 
			cell size. Attempts to size the grid to achieve the desired cell
			size as closely as possible with integer cell counts
		"""

		if (len(bounds) < 4):
			print("Bounds should be suppled as [xOrigin, xMax, yOrigin, yMax]")
			return None

		origin = tuple(bounds[:1])
		xOrigin = bounds[0]
		yOrigin = bounds[2]
		xMax = bounds[1]
		yMax = bounds[3]

		xDist = xMax - xOrigin
		yDist = yMax - yOrigin

		xCellCount = int(round(xDist / cellSize[0]))
		yCellCount = int(round(yDist / cellSize[1]))

		return cls(xDist, yDist, xCellCount, yCellCount, (xOrigin, yOrigin))


	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			return yaml.load(f)


	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes grid parameters to yaml for output to file
			
		"""
		dict_rep = {'xOrigin':data._xOrigin, 'yOrigin':data._yOrigin, 
					'xDistance':data._xDist, 'yDistance':data._yDist, 
					'xCellCount':data._xCellCount, 'yCellCount':data._yCellCount}

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node


	@classmethod
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node)

		init_params = ['xDist', 'yDist', 'xCellCount', 'yCellCount']
		params = [dict_rep[x] for x in init_params]
		origin = (dict_rep['xOrigin'], dict_rep['yOrigin'])
		return cls(*params, origin=origin)

	def save(self, filename):
		with open(filename, 'w') as f:
			yaml.dump(self, f)

	def bin(self, point):
		""" Return indices of cell the point falls into

		"""
		x, y = point

		xRelative = x - self._xOrigin
		yRelative = y - self._yOrigin

		xBin = int(np.floor(xRelative / self._xCellWidth))
		yBin = int(np.floor(yRelative / self._yCellWidth))

		return (xBin, yBin)


	def generateGrid(self):

		xMin = self._xOrigin + self._xCellHalfWidth
		xMax = self._xOrigin + self._xDist - self._xCellHalfWidth

		yMin = self._yOrigin + self._yCellHalfWidth
		yMax = self._yOrigin + self._yDist - self._yCellHalfWidth

		return np.mgrid[xMin:xMax:(self._xCellCount * 1j),
						yMin:yMax:(self._yCellCount * 1j)]


	@property
	def dim(self):
		return (self._xCellCount, self._yCellCount)

	@property
	def edges(self):
		xEdges = np.arange(self._xOrigin, self._xOrigin + self._xDist + 1, self._xCellWidth)
		yEdges = np.arange(self._yOrigin, self._yOrigin + self._yDist + 1, self._yCellWidth)
		return [xEdges, yEdges]

	@property
	def mgrid(self):
		return self.generateGrid()

	@property
	def size(self):
		return self._totalCells