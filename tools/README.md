# Tooling to Build PySAL Meta Package

## Dependencies

- [personal github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
): store it in the file `token`

## Instructions

### Updating package information
- If any new packages have been added to the ecosystem update the `packages` list in `release.yaml` 
- Update relevant data on `start_date` (day after last release), `release_date` (day
  of this release), `version`, and `user` in `release.yaml`

### Updating the changelog
- `make` will run all the steps required to build the new change log

For debugging purposes, the individual steps can be run using:
- `make frozen` will get information about latest package releases
- `make gitcount` will get issues and pulls closed for each package
- `make changelog` will update the change log and write to `changes.md`
  
These require `release_info.py`

### Updating meta packages
- edit the file `changes.md` and incorporate into the release notes on github
- solicit input from devs on highlights for the release notes (on github)
- tag the release
- push the tag
- release on github
