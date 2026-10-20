# Tooling to Build PySAL Meta Package

## Dependencies

- [personal github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
): store it in the file `token`

## Instructions

### Updating package information

#### Adding a package
- Add the package name to the `packages` list in `release_info.py`
- Add the package name (without a version pin) to the `dependencies` list in `../pyproject.toml`
- Add the package name to the appropriate `federation_hierarchy` layer in `../pysal/base.py`
- Add the package name to the `submodules` list in the corresponding layer `__init__.py` (e.g. `../pysal/model/__init__.py`)
- Create a proxy module `../pysal/<layer>/<package>/__init__.py` with `from <package> import *`
- Add the package to all CI environment files in `../ci/`

#### Removing a package
- Remove the package name from the `packages` list in `release_info.py`
- Remove the package from the `dependencies` list in `../pyproject.toml`
- Remove the package from `federation_hierarchy` in `../pysal/base.py`
- Remove the package from the `submodules` list in the corresponding layer `__init__.py`
- Delete the proxy module directory `../pysal/<layer>/<package>/`
- Remove the package from all CI environment files in `../ci/`

### Update the release information
- Update relevant data on `start_date` (day after last release), `release_date` (day
  of this release), `version`, and `user` in `release.yaml`

  If this is a release candidate, do not start the `version` string with `v` but
  do add `rcX` ad the end of the string, where `X` is the number for the current
  release candidate.
  
  If this a production release, the first character in `version` needs to b `v`
  to ensure the publish and release workflow is run in the CI.
  
 

### Updating the changelog
- `make` will run all the steps required to build the new change log

For debugging purposes, the individual steps can be run using:
- `make frozen` will get information about latest package releases
- `make gitcount` will get issues and pulls closed for each package
- `make changelog` will update the change log and write to `changes.md`
  
These require `release_info.py`

### Add and Commit
- `git add ../pyproject.toml`
- `git add release.yaml`
- `git commit -m "REL: <version>"`

 
### Create a tag and push upstream
- `git tag <version>`
- `git push upstream <version>`
  

### Updating meta package release notes
- edit the file `changelog.md` and incorporate into the release notes on github
