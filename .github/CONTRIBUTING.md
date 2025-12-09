# Contributing to PySAL

PySAL moved to the Pull Request model at the 2013 SciPy Conference. Basically, this involves the following components:

1. The organization GitHub account that has the master branch.
2. Releases are made via tags out of master.

A high-level overview of our model is as follows: All work will be submitted via pull requests. Developers will work on branches on their local machines, push these branches to their GitHub repos, and issue a pull request to the organization GitHub account. One of the other developers must review the Pull Request and merge it or, if there are issues, discuss them with the submitter. This ensures developers have a better understanding of the code base and we catch problems before they enter master.

## Initial Setup

1. If you don't have one yet, create your own account on GitHub.
2. Fork `pysal/pysal` into your personal GitHub account.
3. On a laptop/desktop client, clone master from your GitHub account:
   ```bash
   git clone [https://github.com/yourUsername/pysal.git](https://github.com/yourUsername/pysal.git)