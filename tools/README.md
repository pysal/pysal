# Tooling to Build PySAL Meta Package

## Dependencies

- [personal github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
): store it in the file `token`

## Instructions

### Updating package information
- If any new packages have been added to the ecosystem update the `packages` list in `release.yaml` 
- Update relevant data on `start_date` (day after last release), `release_date` (day
  of this release), `version`, and `user` in `release.yaml`

  If this is a release candidate, do not start the `version` string with `v` but
  do add `rcX` ad the end of the string, where `X` is the number for the current
  release candidate.
  
  If this a production release, the first charachter in `version` needs to b `v`
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
- edit the file `changes.md` and incorporate into the release notes on github
