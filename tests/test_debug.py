import sys
import os

def test_debug_paths():
    print("\nCurrent working directory:", os.getcwd())
    print("\nPYTHONPATH:", sys.path)
    print("\nDirectory contents:", os.listdir())
    assert False  # This will force the test to fail and show the debug output 