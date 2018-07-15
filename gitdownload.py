import yaml
import os
import subprocess
import sys

TARGETROOT ="pysal/"


with open('packages.yml') as package_file:
    packages = yaml.load(package_file)
os.system('rm -rf tmp')
os.system('mkdir tmp')
for package in packages:
    #print(package)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        pkgstr = "git clone git@github.com:pysal/{subpackage}.git tmp/{subpackage}".format(subpackage=subpackage)
        print(pkgstr)
        os.system(pkgstr)
