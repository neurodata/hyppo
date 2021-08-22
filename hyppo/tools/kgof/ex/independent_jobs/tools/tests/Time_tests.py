import nose.tools
import unittest

from independent_jobs.tools.Time import Time


class FileSystemTests(unittest.TestCase):

    def test_sec_to_all_11(self):
        secs = 11
        days, hours, minutes, seconds = Time.sec_to_all(secs)
        nose.tools.assert_equal(days, 0)
        nose.tools.assert_equal(hours, 0)
        nose.tools.assert_equal(minutes, 0)
        nose.tools.assert_equal(seconds, 11)
    
    def test_sec_to_all_101(self):
        secs = 101
        days, hours, minutes, seconds = Time.sec_to_all(secs)
        nose.tools.assert_equal(days, 0)
        nose.tools.assert_equal(hours, 0)
        nose.tools.assert_equal(minutes, 1)
        nose.tools.assert_equal(seconds, 41)
    
    def test_sec_to_all_10001(self):
        secs = 10001
        days, hours, minutes, seconds = Time.sec_to_all(secs)
        nose.tools.assert_equal(days, 0)
        nose.tools.assert_equal(hours, 2)
        nose.tools.assert_equal(minutes, 46)
        nose.tools.assert_equal(seconds, 41)
        
    def test_sec_to_all_100001(self):
        secs = 100001
        days, hours, minutes, seconds = Time.sec_to_all(secs)
        nose.tools.assert_equal(days, 1)
        nose.tools.assert_equal(hours, 3)
        nose.tools.assert_equal(minutes, 46)
        nose.tools.assert_equal(seconds, 41)
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
