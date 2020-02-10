# Tooling to Build PySAL Meta Package

## Dependencies

- [yolk3k](https://pypi.org/project/yolk3k/)

## Instructions

- build.py
  - creates a requirements.txt file in this dir that can be hand edited if needed before moving up to `pysal/requirements.txt`
  - creates `pysal/pysal/frozen.py` with information for pinning to subpackage versions
- gitcount-tables.ipynb (WIP): builds changelog for meta package
