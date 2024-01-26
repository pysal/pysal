#!/usr/bin/env python
# coding: utf-8



import pandas
import datetime
import os






today = datetime.datetime.today()




df = pandas.read_csv('r.txt', sep="\t", header=None, names=['Name', 'Type', "Label", 'Date'])




df['Date'] = pandas.to_datetime(df['Date'])




latest = df[df.Type=='Latest'].head(1)




pre = df[df.Type=='Pre-release'].head(1)




if pre.Date.values[0] > latest.Date.values[0]:
    print('rc out')
    rc = int(pre.Label[0].split("c")[1])
else:
    rc = 0
    print('starting rcs')


# generate release (testing)
# rc += 4

tag = f'v{str(today.year)[-2:]}.{today.month:02}'
log_file_path = 'changes.md'
status = ""
if rc <= 4:
    tag = f'{tag}rc{rc}'
    status = "--prerelease"

cmd1 = f"git tag {tag}"
cmd2 = f"git push upstream {tag}"
cmd3 = f"gh release create {tag} --tile '{tag}' --notes-file {log_file_path} {status}"

for cmd in (cmd1, cmd2, cmd3):
    print(cmd)
