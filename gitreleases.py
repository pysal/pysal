import os
import json
import yaml

with open('packages.yml') as package_file:
    packages = yaml.load(package_file)

no_release = []
release = []

for package in packages:
    subpackages = packages[package].split()
    for subpackage in subpackages:
        pkstr = "curl --silent \"https://api.github.com/repos/pysal/{subpackage}/releases/latest\"".format(subpackage=subpackage)
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
            release.append(subpackage)
            #print(tag_name)
            #print(tarball_url)

print("The following {count} packages have git releases:\n\t".format(count=len(release)))
print(release)

print("\n\nThe following {count} packages do not have a git releases:\n\t".format(count=len(no_release)))
print(no_release)



