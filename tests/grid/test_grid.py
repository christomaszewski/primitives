from unittest import TestCase

from primitives.grid import Grid

class GridTest(TestCase):

	def test_grid_construction(self):
		g = Grid(100, 100, 100, 100, (0,0))
		self.assertIsInstance(g, Grid)