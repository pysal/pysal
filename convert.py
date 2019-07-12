import yaml
import os
import subprocess
import sys
import json

# TARGETROOT ="pysal/"

os.system("rm pysal/*.py")

with open("packages.yml") as package_file:
    packages = yaml.load(package_file)

# only tagged packages go in release
with open("tags.json") as tag_file:
    tags = json.load(tag_file)

tagged = list(tags.keys())

print(tagged)

for package in packages:
    com = "rm -fr pysal/{package}".format(package=package)
    os.system(com)
    com = "mkdir pysal/{package}".format(package=package)
    os.system(com)

    # "cp -fr tmp/{subpackage}-master/{subpackage}/* pysal/{package}/".format(package=package, subpackage=subpackage)
    subpackages = packages[package].split()
    for subpackage in subpackages:
        if subpackage == "libpysal":
            com = "cp -rf tmp/{subpackage}/{subpackage}/*  pysal/{package}/".format(
                package=package, subpackage=subpackage
            )
        else:
            com = "cp -rf tmp/{subpackage}/{subpackage} pysal/{package}/{subpackage}".format(
                package=package, subpackage=subpackage
            )
        if subpackage in tagged:
            print(com)
            os.system(com)
        else:
            print("skipping: ", subpackage)


###################
# Rewrite Imports #
###################

def replace(targets, string, replacement):
    c = "find {} -name '*.py' -print | xargs sed -i -- 's/{}/{}/g'".format(targets,
                                                                           string,
                                                                           replacement)
    print(c)
    os.system(c)


replace("pysal/.", "libpysal", "pysal\.lib")
replace("pysal/explore/.", "esda", "pysal\.explore\.esda")
replace("pysal/viz/mapclassify/.", "mapclassify",
        "pysal\.viz\.mapclassify")
replace("pysal/explore/.", "mapclassify", "pysal\.viz\.mapclassify")
replace("pysal/.", "pysal\.spreg", "pysal\.model\.spreg")
replace("pysal/model/spglm/.", "spreg\.", "pysal\.model\.spreg\.")
replace("pysal/model/spglm/.", "from spreg import",
        "from pysal\.model\.spreg import")
replace("pysal/model/spint/.", "from spreg import",
        "from pysal\.model\.spreg import")
replace("pysal/model/spint/.", " spreg\.", " pysal\.model\.spreg\.")
replace("pysal/model/spglm/.", "import spreg\.user_output as USER",
        "from pysal\.model\.spreg import user_output as USER")
replace("pysal/model/spglm/.", "import pysal\.model\.spreg\.user_output as USER",
        "from pysal\.model\.spreg import user_output as USER")
replace("pysal/model/spint/.", "import pysal\.model\.spreg\.user_output as USER",
        "from pysal\.model\.spreg import user_output as USER")
replace("pysal/model/mgwr/.", "import pysal\.model\.spreg\.user_output as USER",
        "from pysal\.model\.spreg import user_output as USER")
replace("pysal/.", "spglm", "pysal\.model\.spglm")
replace("pysal/.", "spvcm", "pysal\.model\.spvcm")
replace("pysal/model/spvcm/.", "def test_val", "\@ut\.skip\\n    def test_val")
replace("pysal/model/spvcm/.", "from spreg", "from pysal\.model\.spreg")
replace("pysal/.", " giddy\.api", "pysal\.explore\.giddy\.api")
replace("pysal/.", "import giddy", "import pysal\.explore\.giddy")
replace("pysal/.", "from giddy",  "from pysal\.explore\.giddy")
replace("pysal/explore/giddy/tests/.", "class Rose_Tester", "@unittest\.skip(\"skipping\")\\nclass Rose_Tester")
replace("pysal/model/mgwr/.", "pysal\.open", "pysal\.lib\.open")
replace("pysal/model/mgwr/.", "pysal\.examples", "pysal\.lib\.examples")
replace("pysal/model/spreg/.", "from spreg", "from pysal\.model\.spreg")
replace("pysal/model/spreg/.", "import spreg", "import pysal\.model\.spreg")
replace("pysal/model/spreg/.", " spreg", " pysal\.model\.spreg")
replace("pysal/model/spvcm/.", "pysal\.examples", "pysal\.lib\.examples")
replace("pysal/model/spvcm/.", "pysal\.queen", "pysal\.lib\.weights\.user\.queen")
replace("pysal/model/spvcm/.", "pysal\.w_subset", "pysal\.lib\.weights\.Wsets\.w_subset")
replace("pysal/viz/splot/.", "from splot\.libpysal", "from pysal\.viz\.splot\.libpysal")
replace("pysal/viz/splot/.", "from splot\.bk", "from pysal\.viz\.splot\.bk")
replace("pysal/viz/splot/.", "from pysal\.viz\.splot\.pysal\.explore\.esda", "from pysal\.viz\.splot\.esda")
replace("pysal/viz/splot/.", "from splot\.pysal\.explore\.esda", "from pysal\.viz\.splot\.pysal\.explore\.esda")
replace("pysal/viz/splot/.", "import pysal as ps", "import pysal")
replace("pysal/viz/splot/.", "ps\.spreg", "pysal\.model\.spreg")
replace("pysal/viz/splot/.", "ps\.lag_spatial", "pysal\.lib\.weights\.spatial_lag\.lag_spatial")
replace("pysal/viz/splot/.", "from esda", "from pysal\.explore\.esda")
replace("pysal/viz/splot/.", "import esda", "import pysal\.explore\.esda as esda")
replace("pysal/viz/splot/.", "import pysal\.esda", "import pysal\.explore\.esda as esda")
replace("pysal/viz/splot/.", "from splot\.esda", "from pysal\.viz\.splot\.esda")
replace("pysal/viz/splot/.", "from spreg", "from pysal\.model\.spreg")
replace("pysal/viz/splot/.", "from splot\.pysal\.lib", "from pysal\.viz\.splot\.libpysal")
replace("pysal/viz/splot/.", "from splot\.giddy", "from pysal\.viz\.splot\.giddy")
replace("pysal/viz/splot/.", "_viz_pysal\.lib_mpl", "_viz_libpysal_mpl")
replace("pysal/viz/splot/.", "import mapclassify", "import pysal\.viz\.mapclassify")
replace("pysal/viz/splot/.", "from splot\.mapping", "from pysal\.viz\.splot\.mapping")
replace("pysal/viz/splot/.", "from splot\._", "from pysal\.viz\.splot\._")
replace("pysal/viz/splot/.", "import spreg", "from pysal\.model import spreg")
replace("pysal/explore/inequality/.", "from inequality", "from pysal\.explore\.inequality")
replace("pysal/model/mgwr/.", "from spreg", "from pysal\.model\.spreg")
replace("pysal/model/mgwr/.", "import spreg", "import pysal\.model\.spreg")
replace("pysal/explore/spaghetti/.", "from spaghetti", "from pysal\.explore\.spaghetti")
replace("pysal/explore/spaghetti/.", "import spaghetti", "import pysal\.explore\.spaghetti")
replace("pysal/explore/segregation/.", "from segregation", "from pysal\.explore\.segregation")
replace("pysal/explore/segregation/.", "import segregation", "import pysal\.explore\.segregation")
replace("pysal/explore/segregation/.", "w_pysal\.lib", "w_libpysal")


init_lines = ["__version__='2.1.0rc'"]
for package in packages:
    os.system("touch pysal/{package}/__init__.py".format(package=package))
    subpackages = packages[package].split()
    if package == "lib":
        pass
    else:
        subpackage_lines = ["from . import {}".format(s) for s in subpackages]
        with open("pysal/{package}/__init__.py".format(package=package), "w") as f:
            f.write("\n".join(subpackage_lines))
    init_lines.append("from . import {}".format(package))

lines = "\n".join(init_lines)

with open("pysal/__init__.py", "w") as outfile:
    outfile.write(lines)
