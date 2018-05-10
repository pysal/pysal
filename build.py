import yaml
import os
import subprocess

GITROOT = "git@github.com:pysal/"
TARGETROOT ="pysal/"


with open('packages.yml') as package_file:
    packages = yaml.load(package_file)

os.system('mkdir tmp')
for package in packages:
    #print(package)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        #print(package, subpackage)
        upstream = GITROOT+subpackage
        #print(upstream)

        # get the zip of the current master branch (or release?) for the subpackage
        pipcom = 'pip download '+subpackage+' --no-binary :all: --no-deps -d tmp'
        tarcom = 'tar xzf tmp/'+subpackage+"*.tar.gz -C tmp; rm tmp/*.gz"

        zipcom = 'unzip tmp/'+subpackage+"*.zip -d tmp"
        cpcom = 'cp -fr tmp/'+subpackage+"*/"+subpackage+" "+"pysal/"+package+"/"
        #print(pipcom)
        os.system(pipcom)
        #print(tarcom)
        tar_res = os.system(tarcom)
        if tar_res != 0:
            zip_res = os.system(zipcom)
            if zip_res != 0:
                print('Neither a zip or tarball is available for %s'% subpackage)
        #print(mvcom)
        os.system(cpcom)

        # pip download subpackage -d tmp --no-deps
        # tar xzf tmp/subpackage*.gz -C tmp
        # mv  tmp/subpackage*/subpackage subparent/

#os.system('git clean -fd')
