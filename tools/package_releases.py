
"""
Grab most recent releases tagged on Github for PySAL subpackages


TODO
 - add logic to get all package releases (not just latest) within the meta-cycle
 - parse the highlights for each release and build up change log highlights for meta
 - build up detailed changes for meta
 - decide if this becomes a workflow
 - consider cron logic to handle release candidate sequence (1-4)
 - consolidate common tooling from releases.py and pakcage_releases.py

"""
import pandas
import os
import json
import urllib
import re
import yaml
from urllib.request import urlopen
from datetime import datetime, timedelta
import requests
import subprocess


packages = ["libpysal", "access", "esda", "giddy", "inequality", "pointpats",
            "segregation", "spaghetti", "mgwr", "momepy", "spglm", "spint", "spreg", "spvcm",
            "tobler", "mapclassify", "splot", "spopt"]


dfs = {}
logs = {}
for package in packages:
    fout = f"{package}.txt"
    cmd = f"gh release list -R pysal/{package} > {fout}"
    print(cmd)
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                   shell=True, check=True)
    df = pandas.read_csv(fout, sep="\t", header=None, names=['Name', 'Type', "Label", 'Date'])
    df['Date'] = pandas.to_datetime(df['Date'])

    dfs[package] = df

    #gh release view -R pysal/segregation
    logfile = f'{package}.chglog'
    cmd = f"gh release view -R pysal/{package} > {logfile}"
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                   shell=True, check=True)
    with open(logfile, 'r') as lf:
        logs[package] = lf.read()


