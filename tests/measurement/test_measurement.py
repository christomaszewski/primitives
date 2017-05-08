from unittest import TestCase

from primitives.measurement import Measurement

class MeasurementTest(TestCase):

	def test_measurement_construction(self):
		m = Measurement((0, 0), (5,5), 0.0)
		self.assertIsInstance(m, Measurement)