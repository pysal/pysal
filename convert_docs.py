import yaml
import os
import subprocess
import sys
import json

#TARGETROOT ="pysal/"


with open('packages.yml') as package_file:
    packages = yaml.load(package_file)


# only tagged packages go in release
with open('tags.json') as tag_file:
    tags = json.load(tag_file)

tagged = list(tags.keys())

print(tagged)

with open("doc/_static/references.bib") as master_bib:
    master = master_bib.readlines()

print(master)

records = []
for package in packages:
    subpackages = packages[package].split()
    for subpackage in subpackages:
        package_bib = "tmp/{subpackage}/doc/_static/references.bib".format(subpackage=subpackage)
        if os.path.isfile(package_bib):
            with open(package_bib) as local_bib:
                local = local_bib.readlines()
                records.extend(local)


with open("doc/_static/references.bib", 'w') as master_bib:
    master_bib.write("".join(records))
