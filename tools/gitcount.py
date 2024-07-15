#!/usr/bin/env python
# coding: utf-8

# # PySAL Change Log Statistics

# urllib3>=1.26
# python-dateutil<=2.8.0
# pytest
# pytest-cov
# coverage## Approach
# - get date of last gh release of each package -> github_released
# - get date of last pypi release of each package -> pypi_released
# - get data of last meta-release -> start_date
# - for each package
#   - get issues between start_date and package_released in master/main
#   - get pulls between start_date and package_released in master/main


import pickle
from release_info import (issues_closed_since, packages,
                          is_pull_request,
                          sorted_by_field,
                          clone_defaults,
                          release_date,
                          start_date,
                          PYSALVER,
                          USER
                          )
import datetime
packages.append('pysal')
clone_defaults(packages)
since = datetime.datetime.combine(start_date, datetime.time(0, 0))
issues = {}
for package in packages:
    issues[package] = issues_closed_since(since, project=f'pysal/{package}')
pulls = {}
for package in packages:
    pulls[package] = issues_closed_since(since, project=f'pysal/{package}',
                                         pulls=True)
pickle.dump(issues, open("issues_closed.p", "wb"))

pickle.dump(pulls, open("pulls_closed.p", "wb"))
