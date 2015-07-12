import os
import pysal

__all__ = ['get_path']

base = os.path.split(pysal.__file__)[0]
example_dir = os.path.join(base,"examples")
dirs = next(os.walk(example_dir))[1]
file_2_dir = {}

for d in dirs:
    tmp = os.path.join(example_dir, d)
    files_in_tmp = os.listdir(tmp)
    for f in files_in_tmp:
        file_2_dir[f] = tmp

def get_path(example_name):
    if example_name in dirs:
        return os.path.join(example_dir,example_name, example_name)
    elif example_name in file_2_dir:
        d = file_2_dir[example_name]
        return os.path.join(d, example_name)
    elif example_name == "":
        return os.path.join(base,'examples', example_name)
    else:
        print(example_name+ ' not found in PySAL built-in examples.')
