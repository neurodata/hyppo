
import os
import unittest

from independent_jobs.tools.Serialization import Serialization


class SerializationTests(unittest.TestCase):

    def test_serialize_object_file_exists(self):
        filename = "temp.bin"
        obj = [1, 2, 3]
        with self.assertRaises(OSError):
            Serialization.serialize_object(obj, filename)
            Serialization.serialize_object(obj, filename)
        
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)
        
    def test_serialize_object_file_not_exists(self):
        filename = "temp.bin"
        obj = [1, 2, 3]
        
        try:
            os.remove(filename)
        except OSError:
            pass
        
        Serialization.serialize_object(obj, filename)
        self.assertTrue(os.path.exists(filename))
        
    def test_serialize_and_deserialize(self):
        filename = "temp.bin"
        obj = [1, 2, 3]
        
        Serialization.serialize_object_overwrite(obj, filename)
        obj2 = Serialization.deserialize_object(filename)
        
        self.assertEqual(obj, obj2)
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
