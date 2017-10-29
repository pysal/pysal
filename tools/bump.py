"""
Update version numbers in a release branch
"""
import datetime
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pysal.version import version

MAJOR, MINOR, MICRO = map(int, version.split("."))
today = datetime.date.today()
year = str(today.year)
day = str(today.day)
month = str(today.month)

version = "{}.{}.{}".format(MAJOR, MINOR, MICRO)
num = {"MAJOR": MAJOR, "MINOR": MINOR, "MICRO": MICRO}

with open('../pysal/version.py') as v_file:
    lines = v_file.readlines()
    v_line = lines[1]
    d_line = lines[2]
    v_line = v_line.split()
    v_line[-1] = '"' + version + '"'
    v_line = (" ").join(v_line)+"\n"
    di = d_line.index('datetime')
    s0 = d_line[di:].strip()
    s1 = s0[:]
    s1 = s1.split()
    s1[0] = s1[0][:-5]+year+","
    s1[1] = month+","
    s1[2] = day+")"
    s1 = " ".join(s1)
    d_line = d_line.replace(s0, s1)
    lines[1] = v_line
    lines[2] = d_line

with open("../pysal/version.py", 'w') as v_file:
    v_file.write("".join(lines))

with open("../setup.py") as s_file:
    lines = s_file.readlines()
    for i, line in enumerate(lines):
        if line[:5] in num:
            v = num[line[:5]]
            line = "{} = {}\n".format(line[:5], v)
            lines[i] = line

with open("../setup.py", 'w') as s_file:
    s_file.write("".join(lines))

with open("../doc/source/conf.py") as c_file:
    lines = c_file.readlines()
    for i, line in enumerate(lines):
        tok = line[:9]
        if tok == 'version =':
            lines[i] = "version = '{}'\n".format(version)
            print(lines[i])
        elif tok == 'release =':
            lines[i] = "release = '{}'\n".format(version)
            print(lines[i])

with open("../doc/source/conf.py", 'w') as c_file:
    c_file.write("".join(lines))

with open("../doc/source/index.rst") as i_file:
    lines = i_file.readlines()
    for i, line in enumerate(lines):
        if line[:13] == '    - `Stable':
            a = '    - `Stable {}'.format(version)
            b = '(Released {}-{}-{})'.format(year, month, day)
            lines[i] = '{} {} <users/installation.html>`_\n'.format(a, b)
        if line[:13] == '    - `Develo':
            lines[i] = ('    - `Development'
                        '  <http://github.com/pysal/pysal/tree/dev>`'
                        '_\n')

with open("../doc/source/index.rst", 'w') as i_file:
    i_file.write("".join(lines))
