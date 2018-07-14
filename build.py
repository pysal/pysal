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
        cpcom = 'cp -fr tmp/'+subpackage+"*/"+subpackage+" "+"pysal/"+package+"/"
        os.system(cpcom)
        print("%s via wget from github" % subpackage)


#os.system('git clean -fd')

# libpysal

os.system('pip download libpysal --no-binary :all: --no-deps -d tmp')
os.system('tar xzf tmp/libpysal*.tar.gz -C tmp; rm tmp/*.gz')
os.system('cp -fr tmp/libpysal*/libpysal/* pysal/lib/.')

# handle libpysal if python=3.+
if sys.version_info[0] == 3:
    twothreecom = '2to3 -nw pysal/lib/'
    print(twothreecom)
    os.system(twothreecom)

"""
# replace all references to libpysal with pysal.lib
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/libpysal/pysal\.lib/g'"
os.system(c)

# replace all references to esda with pysal.explore.esda
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/esda/pysal\.explore\.esda/g'"
os.system(c)

# replace all references to mapclassify with pysal.viz.mapclassify
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/mapclassify/pysal\.viz\.mapclassify/g'"
os.system(c)

# replace all references to .legendgram with pysal.viz.legendgram
#c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/\.legendgram/pysal\.viz\.legendgram/g'"
#os.system(c)

c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/\.vizpysal\./\./g'"
os.system(c)

# replace all references to pysal.spreg with pysal.model.spreg
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/pysal\.spreg/pysal\.model\.spreg/g'"
os.system(c)

# replace all references to spglm with pysal.model.spglm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/spglm/pysal\.model\.spglm/g'"
os.system(c)

# replace all references to .spint with pysal.model.spint
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/spint\./pysal\.model\.spint\./g'"
os.system(c)

# fix libpysal.api now that it has just been clobbered
c = "find pysal/. -name 'api.py' -print | xargs sed -i -- 's/weights\.pysal\.model\.spint/weights\.spintW/g'"
os.system(c)

# replace all references to gwr with pysal.model.gwr
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ gwr / pysal\.model\.gwr/g'"
os.system(c)

# replace all references to spvcm with pysal.model.spvcm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ spvcm / pysal\.model\.spvcm /g'"
os.system(c)

# replace all references to spvcm with pysal.model.spvcm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ spvcm\./ pysal\.model\.spvcm\./g'"
os.system(c)

# replace all references in spglm to spreg with pysal.model.spreg
c = "find pysal/model/spglm/. -name '*.py' -print | xargs sed -i -- 's/ spreg\./ pysal\.model\.spreg\./g'"
os.system(c)

# replace all references in spint to spreg with pysal.model.spreg
c = "find pysal/model/spint/. -name '*.py' -print | xargs sed -i -- 's/ spreg\./ pysal\.model\.spreg\./g'"
os.system(c)

# replace all references in spint to spreg with pysal.model.spreg
c = "find pysal/model/spint/. -name '*.py' -print | xargs sed -i -- 's/from spreg import/from pysal\.model\.spreg import/g'"
os.system(c)
"""


# rewrite pysal/__init__.py at the end

init_lines = [
    ". import lib",
    ". explore import esda",
    ". explore import pointpats",
    ". viz import mapclassify",
    ". viz import legendgram",
    ". dynamics import giddy",
    ". model import spreg",
    ". model import spglm",
    ". model import spint",
    ". model import spvcm",
    ". model import gwr"]

init_lines = [ "from "+line for line in init_lines]
lines = "\n".join(init_lines)
with open("pysal/__init__.py", 'w') as outfile:
    outfile.write(lines)

