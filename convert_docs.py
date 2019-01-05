import yaml
import os
import subprocess
import sys
import json
import pybtex.database

#TARGETROOT ="pysal/"

with open('packages.yml') as package_file:
    packages = yaml.load(package_file)


# only tagged packages go in release
with open('tags.json') as tag_file:
    tags = json.load(tag_file)

tagged = list(tags.keys())


from pybtex.database import BibliographyData, Entry

master_data = BibliographyData( {
    'article-minimal': Entry('article', [
        ('author', 'Leslie B. Lamport'),
        ('title', "blah blah blah"),
        ('journal', "Some outlet"),
        ('year', '1986'),
    ]),
})

# handle duplicates
for package in packages:
    subpackages = packages[package].split()
    for subpackage in subpackages:
        package_bib = "tmp/{subpackage}/doc/_static/references.bib".format(subpackage=subpackage)
        if os.path.isfile(package_bib):
            local = pybtex.database.parse_file(package_bib)
            for entry in local.entries:
                if entry not in master_data.entries:
                    master_data.add_entry(entry, local.entries[entry])
                    print('adding', entry)

with open("doc/_static/references.bib", 'w') as master_bib:
    master_bib.write(master_data.to_string('bibtex'))
