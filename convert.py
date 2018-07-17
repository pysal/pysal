import yaml
import os
import subprocess
import sys

TARGETROOT ="pysal/"

os.system('rm pysal/*.py')

with open('packages.yml') as package_file:
    packages = yaml.load(package_file)



for package in packages:
    com = "rm -fr pysal/{package}".format(package=package)
    os.system(com)
    com = "mkdir pysal/{package}".format(package=package)
    os.system(com)

    #"cp -fr tmp/{subpackage}-master/{subpackage}/* pysal/{package}/".format(package=package, subpackage=subpackage)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        if subpackage == 'libpysal':
            com = "cp -rf tmp/{subpackage}/{subpackage}/*  pysal/{package}/".format(package=package, subpackage=subpackage)
        else:
            com = "cp -rf tmp/{subpackage}/{subpackage} pysal/{package}/{subpackage}".format(package=package, subpackage=subpackage)
        print(com)
        os.system(com)

# replace all references to libpysal with pysal.lib
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/libpysal/pysal\.lib/g'"
os.system(c)

# replace all references to esda with pysal.explore.esda
c = "find pysal/explore/. -name '*.py' -print | xargs sed -i -- 's/esda/pysal\.explore\.esda/g'"
os.system(c)

c = "find pysal/dynamics/. -name '*.py' -print | xargs sed -i -- 's/esda/pysal\.explore\.esda/g'"
os.system(c)

# replace all references to mapclassify with pysal.viz.mapclassify
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/mapclassify/pysal\.viz\.mapclassify/g'"
os.system(c)

# replace all references to pysal.spreg with pysal.model.spreg
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/pysal\.spreg/pysal\.model\.spreg/g'"
os.system(c)

# replace all references in spglm to spreg with pysal.model.spreg
c = "find pysal/model/spglm/. -name '*.py' -print | xargs sed -i -- 's/ spreg\./ pysal\.model\.spreg\./g'"
os.system(c)

# replace all references in spint to spreg with pysal.model.spreg
c = "find pysal/model/spint/. -name '*.py' -print | xargs sed -i -- 's/from spreg import/from pysal\.model\.spreg import/g'"
os.system(c)


# replace all references in spint to spreg with pysal.model.spreg
c = "find pysal/model/spint/. -name '*.py' -print | xargs sed -i -- 's/ spreg\./ pysal\.model\.spreg\./g'"
os.system(c)


# replace all references to spglm with pysal.model.spglm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/spglm/pysal\.model\.spglm/g'"
os.system(c)


# replace all references to spvcm with pysal.model.spvcm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ spvcm / pysal\.model\.spvcm /g'"
os.system(c)

# replace all references to spvcm with pysal.model.spvcm
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ spvcm\./ pysal\.model\.spvcm\./g'"
os.system(c)


# fix giddy
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/ giddy\.api/ pysal\.dynamics\.giddy\.api/g'"
os.system(c)

c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/import giddy/import pysal\.dynamics\.giddy/g'"
os.system(c)

c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/from giddy/from pysal\.dynamics\.giddy/g'"
os.system(c)

# fix gwr
c = "find pysal/model/gwr/. -name '*.py' -print | xargs sed -i -- 's/pysal\.open/pysal\.lib\.open/g'"
os.system(c)


c = "find pysal/model/gwr/. -name '*.py' -print | xargs sed -i -- 's/pysal\.examples/pysal\.lib\.examples/g'"
os.system(c)


# fix spreg

c = "find pysal/model/spreg/. -name '*.py' -print | xargs sed -i -- 's/from spreg/from pysal\.model\.spreg/g'"
os.system(c)

c = "find pysal/model/spreg/. -name '*.py' -print | xargs sed -i -- 's/import spreg/import pysal\.model\.spreg/g'"
os.system(c)

c = "find pysal/model/spreg/. -name '*.py' -print | xargs sed -i -- 's/ spreg/ pysal\.model\.spreg/g'"
os.system(c)

# fix spvcm

c = "find pysal/model/spvcm/. -name '*.py' -print | xargs sed -i -- 's/pysal\.examples/pysal\.lib\.examples/g'"
os.system(c)

#pysal.queen_from_shapefile

c = "find pysal/model/spvcm/. -name '*.py' -print | xargs sed -i -- 's/pysal\.queen/pysal\.lib\.weights\.user\.queen/g'"
os.system(c)



c = "find pysal/model/spvcm/. -name '*.py' -print | xargs sed -i -- 's/pysal\.w_subset/pysal\.lib\.weights\.Wsets\.w_subset/g'"
os.system(c)

# fix splot
#from splot.libpysal import plot_spatial_weights
c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.libpysal/from pysal\.viz\.splot\.libpysal/g'"
os.system(c)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.bk/from pysal\.viz\.splot\.bk/g'"
os.system(c)

#from esda.moran import Moran_Local, Moran, Moran_BV, Moran_Local_BV

#c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from esda/from pysal\.explore\.esda/g'"
#os.system(c)

# from pysal.viz.splot.pysal.explore.esda import 
# from splot.esda
c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from pysal\.viz\.splot\.pysal\.explore\.esda/from pysal\.viz\.splot\.esda/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.pysal\.explore\.esda/from pysal\.viz\.splot\.pysal\.explore\.esda/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/import pysal as ps/import pysal/g'"
os.system(c)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/ps\.spreg/pysal\.model\.spreg/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/ps\.lag_spatial/pysal\.lib\.weights\.spatial_lag\.lag_spatial/g'"
os.system(c)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from esda/from pysal\.explore\.esda/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/import esda/import pysal\.explore\.esda/g'"
os.system(c)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/import pysal\.explore\.esda/import pysal\.explore\.esda as esda/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.esda/from pysal\.viz\.splot\.esda/g'"
os.system(c)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from spreg/from pysal\.model\.spreg/g'"
os.system(c)

#from splot.libpysal import plot_spatial_weights
#from splot.pysal.lib import plot_spatial_weights

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.pysal\.lib/from pysal\.viz\.splot\.libpysal/g'"
os.system(c)


c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/from splot\.giddy/from pysal\.viz\.splot\.giddy/g'"
os.system(c)


#from ._viz_pysal.lib_mpl import (plot_spatial_weights)
#from ._viz_libpysal_mpl import (plot_spatial_weights)

c = "find pysal/viz/splot/. -name '*.py' -print | xargs sed -i -- 's/_viz_pysal\.lib_mpl/_viz_libpysal_mpl/g'"
os.system(c)
#
#
#c = "find pysal/model/gwr/. -name '*.py' -print | xargs sed -i -- 's/pysal\.open/pysal\.lib\.examples\.open/g'"
#os.system(c)


# final change to pysalnext from pysal

c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/pysal/pysalnext/g'"
os.system(c)



"""

c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/\.vizpysal\./\./g'"
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


# replace all references to .legendgram with pysal.viz.legendgram
c = "find pysal/. -name '*.py' -print | xargs sed -i -- 's/\.legendgram/pysal\.viz\.legendgram/g'"
os.system(c)


"""


# rewrite pysal/__init__.py at the end

#init_lines = [
#    ". import lib",
#    ". explore import esda",
#    ". explore import pointpats",
#    ". viz import mapclassify",
#    ". dynamics import giddy",
#    ". model import spreg",
#    ". model import spglm",
#    ". model import spint",
#    ". model import spvcm",
#    ". model import gwr"]
#
#init_lines = [ "from "+line for line in init_lines]
#lines = "\n".join(init_lines)
#with open("pysal/__init__.py", 'w') as outfile:
#    outfile.write(lines)

init_lines = []
for package in packages:
    subpackages = packages[package].split()
    for subpackage in subpackages:
        if subpackage == 'libpysal':
            init_line = "from . import lib"
        else:
            init_line = "from . {package} import {subpackage}".format(package=package, subpackage=subpackage)

        init_lines.append(init_line)

lines = "\n".join(init_lines)

with open("pysal/__init__.py", 'w') as outfile:
    outfile.write(lines)
