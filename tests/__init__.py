import os
_TEST_ROOT = os.path.dirname(__file__) # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT) # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, '../mlsopsbasic/data/archive/images') # root of data

"""
    Just to help with the reusable paths.
    These can help you refer to your data files during testing. For example, in another test file, I could write:
            from tests import _PATH_DATA
    Which then contains the root path to the data.
"""
print(f"From __innit.py file.\nProject root: {_PROJECT_ROOT}\nData root: {_PATH_DATA}")
