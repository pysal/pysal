import yaml
import os
import subprocess
import sys

TARGETROOT ="pysal/"


with open('packages.yml') as package_file:
    packages = yaml.load(package_file)

os.system('mkdir tmp')
for package in packages:
    #print(package)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        """
        #print(package, subpackage)
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
        """
        pkgstr = "https://github.com/pysal/%s/archive/master.zip" % subpackage 
        downloadcommand = "wget %s; mv master.zip tmp/." % pkgstr
        #downloadcommand = "wget https://github.com/pysal/splot/archive/master.zip; mv master.zip tmp/."
        os.system(downloadcommand)
        os.system('unzip tmp/master.zip -d tmp')
        os.system('rm tmp/master.zip')
        print("%s via wget from github" % subpackage)

