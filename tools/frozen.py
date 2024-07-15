#!/usr/bin/env python

import json
from release_info import get_github_info

releases = get_github_info()  

requirements = []
frozen_dict = {}
for package in releases:
    version = releases[package]['version']
    version = version.replace("v","")
    version = version.replace("V","")
    requirements.append(f'{package}>={version}')
    frozen_dict[package] = version

with open('frozen.txt', 'w') as f:
    f.write("\n".join(requirements))

import pickle 
pickle.dump(releases, open( "releases.p", "wb" ) )

