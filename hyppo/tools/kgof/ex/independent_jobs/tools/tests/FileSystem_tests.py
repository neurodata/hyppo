
import os
import tempfile
import unittest

from independent_jobs.tools.FileSystem import FileSystem


class FileSystemTests(unittest.TestCase):

    def test_file_not_exists1(self):
        filename = "./temp.bin"
        try:
            os.remove(filename)
        except OSError:
            pass
        
        self.assertFalse(FileSystem.file_exists_new_shell(filename))
        
    def test_file_exists1(self):
        filename = "./temp.bin"
        f = open(filename, 'w')
        f.close()
        self.assertTrue(FileSystem.file_exists_new_shell(filename))
        
        try:
            os.remove(filename)
        except OSError:
            pass
        
    def test_file_not_exists2(self):
        filename = "temp.bin"
        try:
            os.remove(filename)
        except OSError:
            pass
        
        self.assertFalse(FileSystem.file_exists_new_shell(filename))
        
    def test_file_exists2(self):
        filename = "temp.bin"
        f = open(filename, 'w')
        f.close()
        self.assertTrue(FileSystem.file_exists_new_shell(filename))
        
        try:
            os.remove(filename)
        except OSError:
            pass
        
    def test_get_unique_filename(self):
        for _ in range(100):
            fn = FileSystem.get_unique_filename("")
            self.assertFalse(os.path.exists(fn))
            
    def test_delete_dir_failsafe(self):
        # create dir
        dirname = tempfile.mkdtemp()
        try:
            os.mkdir(dirname)
        except OSError:
            pass
        self.assertTrue(os.path.isdir(dirname))
        
        # put a file to have a non empty dir
        open(os.sep.join([dirname, 'temp']), 'a').close()
        
        # delete and make sure it works
        FileSystem.delete_dir_failsafe(dirname)
        self.assertFalse(os.path.isdir(dirname))
    
    def test_cmd_exists_false(self):
        cmd = "assdjglksdjsdf"
        self.assertFalse(FileSystem.cmd_exists(cmd))
    
    def test_cmd_exists_true(self):
        cmd = "ls"
        self.assertTrue(FileSystem.cmd_exists(cmd))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
