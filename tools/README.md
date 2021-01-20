# Tooling to Build PySAL Meta Package

## Dependencies

- [yolk3k](https://pypi.org/project/yolk3k/)
- [personal github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
): store it in the file `token`

## Instructions

### Updating package information
- If any new packages have been added to the ecosystem update the `packages` list
in `release_info.py`
- Change `USER` in `release_info.py`
- Change `PYSALVER` in `release_info.py`

### Notebooks to run in sequence
- frozen.ipynb creates `frozen.txt` in this dir that can be hand edited if needed before integrating those versions into to `pysal/requirements.txt`
- 100-gitcount.ipynb
  - gets git release information
  - clones releases
- 110-gitcount-tables.ipynb
  - update the the release dates for this (cell 2) and the last (cell 6) meta release
  - builds change log for meta package

### Updating meta packages
- using the `tools/frozen.txt` file, update `./requirements.txt`
- tag the release
- push the tag

