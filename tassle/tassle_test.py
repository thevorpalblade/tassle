import unittest
from .axion_generator import Axion


class TestAxionMethods(unittest.TestCase):
    """A class for unit testing the Axion generator provided by tassle"""

    def setUp(self):
        self.a = Axion()

    def testTest(self):
        self.assertEqual(self.a.v_std, 200)


    def test2Test(self):
        self.assertEqual(self.a.v_std, 200)


if __name__ == '__main__':
    unittest.main()
