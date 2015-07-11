import os
import pysal

__all__ = ['get_path']


base = os.path.split(pysal.__file__)[0]
dirs = next(os.walk('.'))[1][2:]
file_2_dir = {}

for d in dirs:
    tmp = os.path.join(base,'examples',d)
    files_in_tmp = os.listdir(tmp)
    for file in files_in_tmp:
        file_2_dir[file] = tmp


def get_path(example_name):
    if example_name in file_2_dir:
        d = file_2_dir[example_name]
        return os.path.join(d, example_name)
    else:
        print(example_name+ ' not found in examples.')
