"""
Grab most recent releases tagged on Github for PySAL subpackages

"""
import os
import json
import datetime

with open('package_versions.txt', 'r') as package_list:
    packages = dict([line.strip().split()
                     for line in package_list.readlines()])


def get_github_info():
    """
    Get information about subpackage releases that have been tagged on github
    """
    no_release = []
    release = {}

    for package in packages:
        pkstr = f"curl --silent \"https://api.github.com/repos/pysal/{package}/releases/latest\""
        result = os.popen(pkstr).read()
        d = json.loads(result)
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
        release[package]['release_date'] = datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ' )

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
        release_date = datetime.datetime.strptime(release, '%Y-%m-%dT%H:%M:%S')
        releases[package] = {'version': last,
                             'released': release_date}

    return releases



def clone_releases():
    """
    Clone the releases in tmprelease_date
    """
    os.system('rm -rf tmp')
    os.system('mkdir tmp')
    with open('tarballs.json', 'r') as file_name:
        packages = json.load(file_name)
        for package in packages:
            print(package, packages[package]['version'])
            tag = packages[package]['version']
            pkgstr = (
                f'git clone --branch {tag}'
                f' https://github.com/pysal/{package}.git'
                f' tmp/{package}'
                )
            print(pkgstr)
            os.system(pkgstr)



if __name__ == "__main__":
    get_tags()
    clone_releases()
