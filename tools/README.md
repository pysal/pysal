# Tooling to Build PySAL Meta Package

## Dependencies

- [personal github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
): store it in the file `token`

## Instructions

### Updating package information
- If any new packages have been added to the ecosystem update the `packages` list in `release.yaml` 
- Update relevant data on `start_date` (day after last release), `release_date` (day
  of this release), `version`, and `user` in `release.yaml`

### Notebooks to run in sequence
- frozen.ipynb creates `frozen.txt` in this dir that can be hand edited if needed before integrating those versions into to `pysal/requirements.txt`. It also generates `pysal/pysal/frozen.py` which defines the package versions going into this release
- 100-gitcount.ipynb
  - gets git release information
  - clones package repos if new, else git pulls each package
- 110-gitcount-tables.ipynb
  - builds change log for meta package
  - updates the pins for the pysal packages in `../pyproject.toml`

### Updating meta packages
- edit the file `changes.md` and incorporate into the release notes on github
- solicit input from devs on highlights for the release notes (on github)
- tag the release
- push the tag
- release on github
