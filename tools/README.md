# Tooling to Build PySAL Meta Package

## Dependencies

- [yolk3k](https://pypi.org/project/yolk3k/)
- personal token for github api: store it in the file `token`
  https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line

## Instructions

### Updating package information
- If any new packages have been added to the ecosystem update the `packages` list
in `release_info.py`
- Change `USER` in `release_info.py`
- Change `PYSALVER` in `release_info.py`

### Notebooks to run in sequence
- frozen.ipynb creates a requirements.txt file in this dir that can be hand edited if needed before moving up to `pysal/requirements.txt`
- 100-gitcount.ipynb
  - gets git release information
  - clones releases
- 110-gitcount-tables.ipynb
  - builds change log for meta package
  
  
### Updating meta packages
- using the `tools/requirements.txt` file, update `./requirements.txt`
- tag the release
- push the tag

