# Tooling to Build PySAL Meta Package

## Dependencies

- [yolk3k](https://pypi.org/project/yolk3k/)
- personal token for github api: store it in the file `token`
  https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line

## Instructions

- build.py
  - creates a requirements.txt file in this dir that can be hand edited if needed before moving up to `pysal/requirements.txt`
  - creates `pysal/pysal/frozen.py` with information for pinning to subpackage versions
- 100-gitcount.ipynb
  - gets git release information
  - clones releases
- 110-gitcount-tables.ipynb
  - builds change log for meta package
