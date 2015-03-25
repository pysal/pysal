import os
import pysal

__all__ = ['get_path', 'list_examples']


def get_path(example_name):
    base = os.path.split(pysal.__file__)[0]
    return os.path.join(base, 'examples', example_name)

def list_examples():
    base = os.path.split(pysal.__file__)[0]
    files = os.listdir(os.path.join(base, 'examples'))
    exs = set([os.path.splitext(x)[0] for x in files])
    exs.remove('README')
    exs.remove('__init__')
    return exs
