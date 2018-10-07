"""
Grab most recent releases tagged on Github for PySAL subpacakges

TODO
- [x] grab tarballs
- [ ] move tarballs to properly named src directories (target of convert.py)


"""
import os
import json
import yaml
import requests

with open('packages.yml') as package_file:
    packages = yaml.load(package_file)

def get_release_info():
    """
    Get information about subpackage releases that have been tagged on gith
    """
    no_release = []
    release = {} 

    for package in packages:
        subpackages = packages[package].split()
        for subpackage in subpackages:
            pkstr = "curl --silent \"https://api.github.com/repos/pysal/{subpackage}/releases/latest\"".Format(subpackage=subpackage)
            result = os.popen(pkstr).read()
            d = json.loads(result)
            if 'message' in d:
                if d['message']== 'Not Found':
                    print("{subpackage} has no latest release".format(subpackage=subpackage))
                    no_release.append(subpackage)
                else:
                    print('Something else happened')
            else:
                print("{subpackage} has a latest release".format(subpackage=subpackage))
                tag_name = d['tag_name']
                tarball_url = d['tarball_url']
                release[subpackage] = (tag_name, tarball_url)
                #print(tag_name)
                #print(tarball_url)

    print("The following {count} packages have a git release:\n\t".format(count=len(release.keys())))
    print(release.keys())

    print("\n\nThe following {count} packages do not have a git release:\n\t".format(count=len(no_release)))
    print(no_release)

    with open('tarballs.json', 'w') as fp:
        json.dump(release, fp)

def get_tarballs():
    """
    Grab tarballs for releases and put in a temporary directory for furhter processing
    """
    with open('tarballs.json', 'r') as fp:
        sources = json.load(fp)
    os.system('rm -rf tarballs')
    os.system('mkdir tarballs')
    for subpackage in sources.keys():
        print(subpackage)
        url = sources[subpackage][-1]
        print(url)
        target = "tarballs/{pkg}.tar.gz".format(pkg=subpackage)
        print(target)
        resp = requests.get(url)
        with open(target, 'wb') as target_file:
            target_file.write(resp.content)

    return sources
