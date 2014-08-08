.. _installation:

==============
Install  PySAL 
==============


PySAL is built upon the Python scientific stack, primarily numpy and
scipy. While these libraries are packaged for several platforms, the
Anaconda and Enthought Python distributions include them along with the core
Python library. Both Anaconda and Enthought Canopy not only ship numpy and scipy, but PySAL as well! 

- `Anaconda Python distribution <http://continuum.io/downloads.html>`_
- `Enthought Canopy <https://www.enthought.com/downloads>`_


If you do not wish to use either Anaconda or Enthought you should make sure
that make sure the following software packages are available on your machine:

* `Python <http://www.python.org/download>`_ 2.6, or 2.7 
* `numpy <http://new.scipy.org/download.html>`_ 1.3 or later
* `scipy <http://new.scipy.org/download.html>`_ 0.11 or later

Getting your feet wet
----------------------

You can start using PySAL right away on the web with Wakari or
PythonAnywhere. 

wakari http://continuum.io/wakari

PythonAnywhere https://www.pythonanywhere.com/


Download and install 
--------------------

PySAL is available on the `Python Package Index
<http://pypi.python.org/pypi/pysal>`_, which means it can be
downloaded and installed manually or from the command line using 
easy_install, as follows::

 $ easy_install pysal

Alternatively, grab the source distribution (.tar.gz) and decompress it to your selected destination. Open a command shell and navigate to the decompressed pysal folder. Type::

 $ python setup.py install

Windows users can download a graphical installer from
`Sourceforge <http://sourceforge.net/projects/pysal/files/?source=navbar>`_.

Development version on GitHub 
-----------------------------

Developers can checkout PySAL using **git**::

 $ git clone https://github.com/pysal/pysal.git 

Open a command shell and navigate to the cloned pysal
directory. Type::

 $ python setup.py develop

The 'develop' subcommand builds the modules in place 
and modifies sys.path to include the code.
The advantage of this method is that you get the latest code 
but don't have to fuss with editing system environment variables.

To test your setup, start a Python session and type::

 >>> import pysal

Keep up to date with pysal development by 'pulling' the latest
changes::

 $ git pull

Windows
~~~~~~~~

To keep up to date with PySAL development, you will need a Git client that allows you to access and 
update the code from our repository. We recommend 
`GitHub Windows <http://windows.github.com/>`_ for a more graphical client, or
`Git Bash <https://code.google.com/p/msysgit/downloads/list?q=label:Featured>`_ for a
command line client. This one gives you a nice Unix-like shell with
familiar commands. Here is `a nice tutorial
<http://openhatch.org/missions/windows-setup/>`_ on getting going with Open
Source software on Windows. 

After cloning pysal, install it in develop mode so Python knows where to find it. 

Open a command shell and navigate to the cloned pysal
directory. Type::

 $ python setup.py develop

To test your setup, start a Python session and type::

 >>> import pysal

Keep up to date with pysal development by 'pulling' the latest
changes::

 $ git pull


Troubleshooting
===============

If you experience problems when building, installing, or testing pysal, ask for
help on the
`OpenSpace <http://geodacenter.asu.edu/support/community>`_ 
list or
browse the archives of the
`pysal-dev <http://groups.google.com/group/pysal-dev?pli=1>`_ 
google group. 

Please include the output of the following commands in your message:

1) Platform information::

    python -c 'import os,sys;print os.name, sys.platform'
    uname -a

2) Python version::
    
    python -c 'import sys; print sys.version'

3) SciPy version::

    python -c 'import scipy; print scipy.__version__'

3) NumPy version::

    python -c 'import numpy; print numpy.__version__'

4) Feel free to add any other relevant information.
   For example, the full output (both stdout and stderr) of the pysal
   installation command can be very helpful. Since this output can be
   rather large, ask before sending it into the mailing list (or
   better yet, to one of the developers, if asked).




