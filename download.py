import yaml
import os
import subprocess
import sys



with open('packages.yml') as package_file:
    packages = yaml.load(package_file)
os.system('rm -rf tmp')
os.system('mkdir tmp')
for package in packages:
    #print(package)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        pkgstr = "https://github.com/pysal/%s/archive/master.zip" % subpackage 
        downloadcommand = "wget %s; mv master.zip tmp/." % pkgstr
        #downloadcommand = "wget https://github.com/pysal/splot/archive/master.zip; mv master.zip tmp/."
        os.system(downloadcommand)
        os.system('unzip tmp/master.zip -d tmp')
        os.system('rm tmp/master.zip')
        print("%s via wget from github" % subpackage)

# get libpysal from master

pkgstr = "https://github.com/pysal/libpysal/archive/master.zip"
downloadcommand = "wget %s; mv master.zip tmp/." % pkgstr
os.system(downloadcommand)
os.system('unzip tmp/master.zip -d tmp')
os.system('rm tmp/master.zip')
