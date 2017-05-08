from unittest import TestCase

import heapq

from primitives.track import Track, TrackState

class TrackTest(TestCase):

	def setUp(self):
		pass

	def generateRandomTrack(self):
		pass


	def test_track_construction(self):
		# Initialize empty track
		t = Track()

		# Check state has been initialized to EMPTY
		self.assertEqual(t.state, TrackState.EMPTY)

		# Check size, age, and length return 0
		self.assertEqual(t.size(), 0)
		self.assertEqual(t.age(), 0)
		self.assertEqual(t.length(), 0)

		# Check lastSeen and endPoint return None
		self.assertIsNone(t.lastSeen)
		self.assertIsNone(t.endPoint)

		# Add observation to Track
		point = (0.0, 0.0)
		time = 0.0
		t.addObservation(point, time)

		# Check state has updated to ACTIVE
		self.assertEqual(t.state, TrackState.ACTIVE)

		# Check size, age, and length return 1, 0, and 0
		self.assertEqual(t.size(), 1)
		self.assertEqual(t.age(), 0)
		self.assertEqual(t.length(), 0)

		# Check lastSeen and endPoint refer to point and time
		self.assertEqual(t.lastSeen, time)
		self.assertEqual(t.endPoint, point) 


	def test_measurement_construction(self):
		# Initialize track at origin
		t = Track((0.0,0.0), 0.0)

		# Check that state has been initialized to active
		self.assertEqual(t.state, TrackState.ACTIVE)

		# Add a second point to the track
		point = (5.0, 3.0)
		time = 2.0
		t.addObservation(point, time)

		# Calculate measurements using various Methods and scoring
		midpointTimeM = t.measureVelocity(method='midpoint', scoring='time')[0]
		firstLengthM = t.measureVelocity(method='first', scoring='length')[0]
		lastDispM = t.measureVelocity(method='last', scoring='displacement')[0]
		midpointCompositeM = t.measureVelocity(scoring='composite')[0]

		# Check method for measurement positioning
		self.assertAlmostEqual(midpointTimeM.point[0], 2.5)
		self.assertAlmostEqual(midpointTimeM.point[1], 1.5)

		self.assertAlmostEqual(firstLengthM.point[0], 0.0)
		self.assertAlmostEqual(firstLengthM.point[1], 0.0)

		self.assertAlmostEqual(lastDispM.point[0], 5.0)
		self.assertAlmostEqual(lastDispM.point[1], 3.0)

		# Check measurement scoring
		self.assertAlmostEqual(midpointTimeM.score, time)
		self.assertAlmostEqual(firstLengthM.score, 2)
		self.assertAlmostEqual(lastDispM.score, 5.83095189485)
		#self.assertAlmostEqual(midpointCompositeM.score, 2.38309518948)

		# Check computed velocity value
		self.assertAlmostEqual(midpointTimeM.value[0], 2.5)
		self.assertAlmostEqual(midpointTimeM.value[1], 1.5)