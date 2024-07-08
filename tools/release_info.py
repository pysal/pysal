"""
Grab most recent releases tagged on Github for PySAL subpackages


TODO
- [ ] update dependencies in pyproj.toml for pinning new releases of
pysal packages

"""

import os
import subprocess
import json
import urllib
import re
import yaml
from urllib.request import urlopen
from datetime import datetime, timedelta
import requests


with open("release.yaml", "r") as stream:
    info = yaml.safe_load(stream)

release_date = info["release_date"]
PYSALVER = info["version"]
start_date = info["start_date"]
USER = info["user"]


ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100
element_pat = re.compile(r"<(.+?)>")
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')


# get github token:
with open("token", "r") as token_file:
    token = token_file.read().strip()

gh_session = requests.Session()
gh_session.auth = (USER, token)


packages = [
    "libpysal",
    "access",
    "esda",
    "giddy",
    "inequality",
    "pointpats",
    "segregation",
    "spaghetti",
    "mgwr",
    "momepy",
    "spglm",
    "spint",
    "spreg",
    "spvcm",
    "tobler",
    "mapclassify",
    "splot",
    "spopt",
]


def get_github_info(packages=packages):
    """
    Get information about subpackage releases that have been tagged on github
    """
    no_release = []
    release = {}

    for package in packages:
        url = f"https://api.github.com/repos/pysal/{package}/releases/latest"
        print(url)
        d = json.loads(gh_session.get(url).text)
        if "message" in d:
            if d["message"] == "Not Found":
                print(f"{package} has no latest release")
                no_release.append(package)
            else:
                print("Something else happened")
                print(d)
        else:
            tag_name = d["tag_name"]
            tarball_url = d["tarball_url"]
            release[package] = {
                "version": tag_name,
                "url": tarball_url,
                "release_date": d["published_at"],
            }

    with open("tarballs.json", "w") as fp:
        json.dump(release, fp)

    for package in release:
        dt = release[package]["release_date"]
        date_format = "%Y-%m-%dT%H:%M:%SZ"
        release_date = datetime.strptime(dt, date_format)
        release[package]["release_date"] = release_date

    return release


def get_pypi_info():
    """
    Get information about subpackage releases that have been tagged on pypi
    """
    releases = {}
    for package in packages:
        url = f"https://pypi.python.org/pypi/{package}/json"
        data = json.load(urllib.request.urlopen(url))
        keys = list(data["releases"].keys())
        last = keys[-1]
        release = data["releases"][last][0]["upload_time"]
        release_date = datetime.strptime(release, "%Y-%m-%dT%H:%M:%S")
        releases[package] = {"version": last, "released": release_date}

    return releases


def clone_masters():
    clone_releases(tag="master")


def clone_mains():
    clone_releases(tag="main")


def clone_defaults(packages=packages, cwd=os.getcwd()):
    for package in packages:
        directory_path = f"tmp/{package}"

        # if already cloned, we pull, otherwise clone
        if os.path.isdir(directory_path):
            print(f"{directory_path} exists, git pull required")
            os.chdir(directory_path)
            result = subprocess.run(["git", "pull"],
                                    capture_output=True, text=True)
            print("Output:\n", result.stdout)
            print("Errors:\n", result.stderr)
            os.chdir(cwd)
        else:
            url = f"https://api.github.com/repos/pysal/{package}"
            data = json.load(urllib.request.urlopen(url))
            branch = data["default_branch"]
            pkgstr = (
                f"git clone --branch {branch}"
                f" https://github.com/pysal/{package}.git"
                f" tmp/{package}"
            )
            print(pkgstr)
            os.system(pkgstr)


def clone_releases(tag=None):
    """
    Clone the releases in tmprelease_date
    """
    os.system("rm -rf tmp")
    os.system("mkdir tmp")
    for package in packages:
        print(package, packages[package])
        if tag:
            branch = tag
        else:
            branch = packages[package]
        pkgstr = (
            f"git clone --branch {branch}"
            f" https://github.com/pysal/{package}.git"
            f" tmp/{package}"
        )
        print(pkgstr)
        os.system(pkgstr)


def parse_link_header(headers):
    link_s = headers.get("link", "")
    urls = element_pat.findall(link_s)
    rels = rel_pat.findall(link_s)
    d = {}
    for rel, url in zip(rels, urls):
        d[rel] = url
    return d


def get_paged_request(url):
    """get a full list, handling APIv3's paging"""
    results = []
    while url:
        # print("fetching %s" % url, file=sys.stderr)
        f = urlopen(url)
        results.extend(json.load(f))
        links = parse_link_header(f.headers)
        url = links.get("next")
    return results


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
        idict[i["number"]] = i
    return idict


def get_url(url):
    d = json.loads(gh_session.get(url).text)
    return d


def get_issues(project="pysal/pysal", state="closed", pulls=False):
    """Get a list of the issues from Github api"""
    which = "pulls" if pulls else "issues"
    url = f"https://api.github.com/repos/{project}/{which}?state={state}"
    return get_url(url)


def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return "pull_request_url" in issue


def issues_closed_since(
    period=timedelta(days=365), project="pysal/pysal", pulls=False
):
    """Get all issues closed since a particular point in time. period
    can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present."""

    which = "pulls" if pulls else "issues"

    if isinstance(period, timedelta):
        period = datetime.now() - period

    url = (
        "https://api.github.com/repos/{}/{}?state=closed&sort=updated&since={}"
        "&per_page={}".format(
            project, which, period.strftime(ISO8601), PER_PAGE
        )
    )

    allclosed = get_url(url)
    filtered = [
        i for i in allclosed if _parse_datetime(i["closed_at"]) > period
    ]

    # exclude rejected PRs
    if pulls:
        filtered = [pr for pr in filtered if pr["merged_at"]]

    return filtered


def sorted_by_field(issues, field="closed_at", reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key=lambda i: i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title."""
    # titles may have unicode in them, so we must encode everything below
    if show_urls:
        for i in issues:
            role = "ghpull" if "merged_at" in i else "ghissue"
            title = i["title"].encode("utf-8")
            print(f"* :{role}:`{i['number']}`: {title}")
    else:
        for i in issues:
            title = i["title"].encode("utf-8")
            print(f"* {i['number']}: {title}")


def get_meta_releases():
    url = "https://api.github.com/repos/pysal/pysal/releases"
    return get_url(url)
