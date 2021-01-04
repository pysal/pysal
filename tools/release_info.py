"""
Grab most recent releases tagged on Github for PySAL subpackages

"""
import os
import json
import urllib
import re
from urllib.request import urlopen
from datetime import datetime, timedelta
import requests

PYSALVER = '2.3.0'

USER = "sjsrey"

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100
element_pat = re.compile(r'<(.+?)>')
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')



# get github token:
with open('token', 'r') as token_file:
    token = token_file.read().strip()

gh_session = requests.Session()
gh_session.auth = (USER, token)

with open('package_versions.txt', 'r') as package_list:
    packages = dict([line.strip().split()
                     for line in package_list.readlines()])

packages['pysal'] = PYSALVER

def get_github_info():
    """
    Get information about subpackage releases that have been tagged on github
    """
    no_release = []
    release = {}

    for package in packages:
        url = f"https://api.github.com/repos/pysal/{package}/releases/latest"
        print(url)
        d = json.loads(gh_session.get(url).text)
        if 'message' in d:
            if d['message'] == 'Not Found':
                print(f"{package} has no latest release")
                no_release.append(package)
            else:
                print('Something else happened')
        else:
            tag_name = d['tag_name']
            tarball_url = d['tarball_url']
            #release_date = datetime.datetime.strptime(d['published_at'], '%Y-%m-%dT%H:%M:%SZ')
            release[package] = {'version': tag_name,
                                'url': tarball_url,
                                'release_date': d['published_at']}

    with open('tarballs.json', 'w') as fp:
        json.dump(release, fp)

    for package in release:
        dt = release[package]['release_date']
        release[package]['release_date'] = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ' )

    return release


def get_pypi_info():
    """
    Get information about subpackage releases that have been tagged on pypi
    """
    no_release = []
    releases = {}
    for package in packages:
        url = f"https://pypi.python.org/pypi/{package}/json"
        data = json.load(urllib.request.urlopen(url))
        keys = list(data['releases'].keys())
        last = keys[-1]
        release = data['releases'][last][0]['upload_time']
        release_date = datetime.strptime(release, '%Y-%m-%dT%H:%M:%S')
        releases[package] = {'version': last,
                             'released': release_date}


    return releases



def clone_masters():
    clone_releases(tag='master')

def clone_mains():
    clone_releases(tag='main')

def clone_defaults():
    for package in packages:
        url = f"https://api.github.com/repos/pysal/{package}"
        data = json.load(urllib.request.urlopen(url))
        branch = data['default_branch']
        pkgstr = (
            f'git clone --branch {branch}'
            f' https://github.com/pysal/{package}.git'
            f' tmp/{package}'
            )
        print(pkgstr)
        os.system(pkgstr)


def clone_releases(tag=None):
    """
    Clone the releases in tmprelease_date
    """
    os.system('rm -rf tmp')
    os.system('mkdir tmp')
    for package in packages:
        print(package, packages[package])
        if  tag:
            branch = tag
        else:
            branch = packages[package]
        pkgstr = (
            f'git clone --branch {branch}'
            f' https://github.com/pysal/{package}.git'
            f' tmp/{package}'
            )
        print(pkgstr)
        os.system(pkgstr)



def parse_link_header(headers):
    link_s = headers.get('link', '')
    urls = element_pat.findall(link_s)
    rels = rel_pat.findall(link_s)
    d = {}
    for rel,url in zip(rels, urls):
        d[rel] = url
    return d

def get_paged_request(url):
    """get a full list, handling APIv3's paging"""
    results = []
    while url:
        #print("fetching %s" % url, file=sys.stderr)
        f = urlopen(url)
        results.extend(json.load(f))
        links = parse_link_header(f.headers)
        url = links.get('next')
    return results

def get_issues(project="pysal/pysal", state="closed", pulls=False):
    """Get a list of the issues from the Github API."""
    which = 'pulls' if pulls else 'issues'
    url = "https://api.github.com/repos/%s/%s?state=%s&per_page=%i" % (project, which, state, PER_PAGE)
    return get_paged_request(url)


def _parse_datetime(s):
    """Parse dates in the format returned by the Github API."""
    if s:
        return datetime.strptime(s, ISO8601)
    else:
        return datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    idict = {}
    for i in issues:
        idict[i['number']] = i
    return idict


def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return 'pull_request_url' in issue


def issues_closed_since(period=timedelta(days=365), project="pysal/pysal", pulls=False):
    """Get all issues closed since a particular point in time. period
can either be a datetime object, or a timedelta object. In the
latter case, it is used as a time before the present."""

    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        period = datetime.now() - period
    url = "https://api.github.com/repos/%s/%s?state=closed&sort=updated&since=%s&per_page=%i" % (project, which, period.strftime(ISO8601), PER_PAGE)
    allclosed = get_paged_request(url)
    # allclosed = get_issues(project=project, state='closed', pulls=pulls, since=period)
    filtered = [i for i in allclosed if _parse_datetime(i['closed_at']) > period]

    # exclude rejected PRs
    if pulls:
        filtered = [ pr for pr in filtered if pr['merged_at'] ]

    return filtered


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key = lambda i:i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title.
    """
    # titles may have unicode in them, so we must encode everything below
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            print('* :%s:`%d`: %s' % (role, i['number'],
                                        i['title'].encode('utf-8')))
    else:
        for i in issues:
            print('* %d: %s' % (i['number'], i['title'].encode('utf-8')))



