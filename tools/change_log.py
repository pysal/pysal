#!/usr/bin/env python
# coding: utf-8

# ## PySAL Change Log Statistics: Table Generation

from __future__ import print_function
from collections import Counter
from datetime import date, datetime, time
from datetime import datetime
from release_info import get_pypi_info, get_github_info, clone_masters
import pickle
import release_info
from release_info import release_date, start_date, PYSALVER
import os
import json
import re
import sys
import pandas
import subprocess
from subprocess import check_output

# import yaml
from datetime import datetime, timedelta, time

from dateutil.parser import parse
import pytz

utc = pytz.UTC

try:
    from urllib import urlopen
except:
    from urllib.request import urlopen

since = datetime.combine(start_date, time(0, 0))
CWD = os.path.abspath(os.path.curdir)

with open('frozen.txt', 'r') as package_list:
    packages = package_list.readlines()
    packages = dict([package.strip().split(">=") for package in packages])

packages['pysal'] = release_info.PYSALVER
issues_closed = pickle.load(open("issues_closed.p", 'rb'))
pulls_closed = pickle.load(open('pulls_closed.p', 'rb'))
github_releases = pickle.load(open("releases.p", 'rb'))
ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
pysal_date = release_date

# Create a date object
date_obj = release_date
# Create a time object (optional, default is midnight if not specified)
time_obj = time(0, 0)
# Combine date and time to create a datetime object
datetime_obj = datetime.combine(date_obj, time_obj)
pysal_rel = {'version': f'v{PYSALVER}',
             'release_date': datetime_obj}
github_releases['pysal'] = pysal_rel


final_pulls = {}
final_issues = {}
for package in packages:
    filtered_issues = []
    filtered_pulls = []
    released = github_releases[package]['release_date']
    package_pulls = pulls_closed[package]
    package_issues = issues_closed[package]
    for issue in package_issues:
        # print(issue['number'], issue['title'], issue['closed_at'])
        closed = datetime.strptime(issue['closed_at'], ISO8601)
        if closed <= released and closed > since:
            filtered_issues.append(issue)
    final_issues[package] = filtered_issues
    for pull in package_pulls:
        # print(pull['number'], pull['title'], pull['closed_at'])
        closed = datetime.strptime(pull['closed_at'], ISO8601)
        if closed <= released and closed > since:
            filtered_pulls.append(pull)
    final_pulls[package] = filtered_pulls

issue_details = final_issues
pull_details = final_pulls
github_releases['pysal']['release_date'] = release_date


# skip packages not released since last meta release
# handle meta
mrd = github_releases['pysal']['release_date']
github_releases['pysal']['release_date'] = datetime.combine(mrd, time(0, 0))

for package in github_releases:
    if github_releases[package]['release_date'] > since:
        print("new: ", package)
    else:
        print('old:', package)

since_date = '--since="{start}"'.format(start=start_date.strftime("%Y-%m-%d"))

# commits
cmd = ['git', 'log', '--oneline', since_date]

activity = {}
total_commits = 0
tag_dates = {}
ncommits_total = 0
for subpackage in packages:
    released = github_releases[subpackage]['release_date']
    tag_date = released.strftime("%Y-%m-%d")
    tag_dates[subpackage] = tag_date
    print(tag_date)
    # tag_date = tag_dates[subpackage]
    ncommits = 0
    if released > since:
        os.chdir(CWD)
        os.chdir('tmp/{subpackage}'.format(subpackage=subpackage))
        cmd_until = cmd + ['--until="{tag_date}"'.format(tag_date=tag_date)]
        ncommits = len(check_output(cmd_until).splitlines())
        ncommits_total = len(check_output(cmd).splitlines())
    print(subpackage, ncommits_total, ncommits, tag_date)
    total_commits += ncommits
    activity[subpackage] = ncommits


cmd = ['git', 'log', '--oneline', since_date]

activity = {}
total_commits = 0
for subpackage in packages:
    ncommits = 0
    tag_date = tag_dates[subpackage]
    released = github_releases[subpackage]['release_date']
    if released > since:
        os.chdir(CWD)
        os.chdir('tmp/{subpackage}'.format(subpackage=subpackage))
        cmd_until = cmd + ['--until="{tag_date}"'.format(tag_date=tag_date)]
        ncommits = len(check_output(cmd_until).splitlines())
        print(ncommits)
        ncommits_total = len(check_output(cmd).splitlines())
        print(subpackage, ncommits_total, ncommits, tag_date)
    total_commits += ncommits
    activity[subpackage] = ncommits


identities = {'Levi John Wolf': ('ljwolf', 'Levi John Wolf'),
              'Serge Rey': ('Serge Rey', 'Sergio Rey', 'sjsrey', 'serge'),
              'Wei Kang': ('Wei Kang', 'weikang9009'),
              'Dani Arribas-Bel': ('Dani Arribas-Bel', 'darribas'),
              'Antti Härkönen': ('antth', 'Antti Härkönen', 'Antti Härkönen', 'Antth'),
              'Juan C Duque': ('Juan C Duque', "Juan Duque"),
              'Renan Xavier Cortes': ('Renan Xavier Cortes', 'renanxcortes', 'Renan Xavier Cortes'),
              'Taylor Oshan': ('Tayloroshan', 'Taylor Oshan', 'TaylorOshan'),
              'Tom Gertin': ('@Tomgertin', 'Tom Gertin', '@tomgertin')
              }


def regularize_identity(string):
    string = string.decode()
    for name, aliases in identities.items():
        for alias in aliases:
            if alias in string:
                string = string.replace(alias, name)
    if len(string.split(' ')) > 1:
        string = string.title()
    return string.lstrip('* ')


author_cmd = ['git', 'log', '--format=* %aN', since_date]


author_cmd.append('blank')


authors_global = set()
authors = {}
global_counter = Counter()
counters = dict()
cmd = ['git', 'log', '--oneline', since_date]
total_commits = 0
activity = {}
for subpackage in packages:
    ncommits = 0
    released = github_releases[subpackage]['release_date']
    if released > since:
        os.chdir(CWD)
        os.chdir('tmp/{subpackage}'.format(subpackage=subpackage))
        ncommits = len(check_output(cmd).splitlines())
        tag_date = tag_dates[subpackage]
        tag_date = (datetime.strptime(tag_date, '%Y-%m-%d') +
                    timedelta(days=1)).strftime('%Y-%m-%d')
        author_cmd[-1] = '--until="{tag_date}"'.format(tag_date=tag_date)
        # cmd_until = cmd + ['--until="{tag_date}"'.format(tag_date=tag_date)]
        print(subpackage, author_cmd)

        all_authors = check_output(author_cmd).splitlines()
        counter = Counter([regularize_identity(author)
                          for author in all_authors])
        global_counter += counter
        counters.update({subpackage: counter})
        unique_authors = sorted(set(all_authors))
        authors[subpackage] = unique_authors
        authors_global.update(unique_authors)
    total_commits += ncommits
    activity[subpackage] = ncommits


def get_tag(title, level="##", as_string=True):
    words = title.split()
    tag = "-".join([word.lower() for word in words])
    heading = level+" "+title
    line = "\n\n<a name=\"{}\"></a>".format(tag)
    lines = [line]
    lines.append(heading)
    if as_string:
        return "\n".join(lines)
    else:
        return lines


subs = issue_details.keys()
table = []
txt = []
lines = get_tag("Changes by Package", as_string=False)

for sub in github_releases:
    total = issue_details[sub]
    pr = pull_details[sub]

    row = [sub, activity[sub], len(total), len(pr)]
    table.append(row)
    # line = "\n<a name=\"{sub}\"></a>".format(sub=sub)
    # lines.append(line)
    # line = "### {sub}".format(sub=sub)
    # lines.append(line)
    sub_lower = sub.lower()
    sub_version = github_releases[sub_lower]['version']
    print(f'{sub_lower}, {sub_version}')
    title = f'{sub_lower} {sub_version}'
    lines.extend(get_tag(title, "###", as_string=False))
    for issue in total:
        url = issue['html_url']
        title = issue['title']
        number = issue['number']
        line = "* [#{number}:]({url}) {title} ".format(title=title,
                                                       number=number,
                                                       url=url)
        lines.append(line)


os.chdir(CWD)

df = pandas.DataFrame(
    table, columns=['package', 'commits', 'total issues', 'pulls'])

df.sort_values(['commits', 'pulls'], ascending=False)\
  .to_html('./commit_table.html', index=None)

contributor_table = pandas.DataFrame.from_dict(
    counters).fillna(0).astype(int).T

contributor_table.to_html('./contributor_table.html')

totals = contributor_table.sum(axis=0).T
totals.sort_index().to_frame('commits')

totals = contributor_table.sum(axis=0).T
totals.sort_index().to_frame('commits').to_html('./commits_by_person.html')

n_commits = df.commits.sum()
n_issues = df['total issues'].sum()
n_pulls = df.pulls.sum()

line = ('Overall, there were {n_commits} commits that closed {n_issues} issues'
        ' since our last release'
        ' on {since_date}.\n'.format(n_commits=n_commits, n_issues=n_issues,
                                     since_date=start_date))


with open('changes.md', 'w') as fout:
    fout.write(line)
    fout.write("\n".join(lines))
    fout.write(get_tag("Contributors"))
    fout.write(
        "\n\nMany thanks to all of the following individuals who contributed to this release:\n\n")

    totals = contributor_table.sum(axis=0).T
    contributors = totals.index.values
    contributors.sort()
    contributors = contributors.tolist()
    contributors = [f'\n - {contributor}' for contributor in contributors]
    fout.write("".join(contributors))


df.head()


# Update ../pyproject.toml for minimum pysal package pinning
# get version numbers from frozen.txt
with open('frozen.txt', 'r') as frozen:
    packages = [line.rstrip() for line in frozen.readlines()]

# search pyproject.toml for lines containing package
with open('../pyproject.toml', 'r') as project:
    lines = [line.rstrip() for line in project.readlines()]

# split line ->"    package",  ">=",  "version",
# replace version and rebuild line to update
for package in packages:
    name, version = package.split(">=")
    i, match = [(i, line) for i, line in enumerate(lines) if name in line][0]
    old_name, old_version = match.split(">=")
    new_line = ">=".join([old_name, version+'",'])
    lines[i] = new_line

# write out new pyproject.toml file
with open("../pyproject.toml", 'w') as output:
    output.write("\n".join(lines))
