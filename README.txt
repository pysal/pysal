Dependencies
=============================================================================
Before installing PySAL, make sure the following libraries are properly
installed on you machine.

==Required==
 * Python 2.5 or later
 * Numpy 1.3 
 * Scipy 0.7 or later

Dependencies to build documentation:

* Sphinx
* numpydoc extension to Sphinx (NOT included in EPD)

==Optional but recommended==

 * iPython
 * rtree (NOT included in EPD)

==Enthought Python Distribution (EPD)==

Instead of installing them one by one, you can get all the required
dependencies as well as iPython and a whole bunch of other Python packages by
easily installing the [http://www.enthought.com/products/epd.php Enthought
Python Distribution], available free of charge for Academics. Download it
here:

    http://www.enthought.com/products/edudownload.php. 

=Installing from source code=

As PySAL is currently in alpha stage, we are not yet preparing binary
releases. Users can grab the source code from our subversion repository.

==Linux and Mac OS X (and most `*`Nix machines)==

To download the code, open up a terminal (`/Applications/Utilities/Terminal`)
and move to the directory where you wish to download PySAL by typing:

    cd /path_to_desired/folder

Once there, type the the following command:

    svn checkout http://pysal.googlecode.com/svn/trunk/ pysal-read-only

Note: mind there must be a space between 'trunk/' and 'pysal-read-only'.

This will create a folder called 'pysal-read-only' containing all the source
code into the folder you chose and will allow you to easily update any change
that is made to the code by the developer team. Since PySAL is in active and
intense development, a lot of these changes are often introduced. For this
reason it is preferable to 'tell' Python to look for PySAL in that folder
rather than properly install it as a package. You can do this by adding the
PySAL folder to the Python path. Open the bash profile (if it doesn't already
exist, just create a new text file in the home directory and name it
`.bash_profile`) by typing in the terminal:

    open ~/.bash_profile

Note: replace the command `open` by that of a text editor if you are in Linux
(`gedit` for instance, if you are in Ubuntu).
Now add the following line at the end of the text file:

    export PYTHONPATH=${PYTHONPATH}:"/path_to_desired/folder/pysal-read-only/"

Save and quit the file. Source the bash profile again:

    source ~/.bash_profile

You are all set!!! Now you can open up a fresh python session and start
enjoying PySAL, you should be able to do (within a python session):

 In [1]: import pysal

 In [2]: pysal.open.check()
 PySAL File I/O understands the following file extensions:
 Ext: '.shp', Modes: ['r', 'wb', 'w', 'rb']
 Ext: '.shx', Modes: ['r', 'wb', 'w', 'rb']
 Ext: '.geoda_txt', Modes: ['r']
 Ext: '.dbf', Modes: ['r', 'w']
 Ext: '.gwt', Modes: ['r']
 Ext: '.gal', Modes: ['r', 'w']
 Ext: '.csv', Modes: ['r']
 Ext: '.wkt', Modes: ['r']
 In [3]: 
 
==Windows==

To be able to use PySAL, you will need a SVN client that allows you to access,
download and update the code from our repository. We recommend to use
TortoiseSVN (http://tortoisesvn.tigris.org/), which is free and very easy to
install. The following instructions assume you are using it.

First, create a folder where you want to store PySAL's code. For the sake of this
example, we will name it `PySALsvn` and put it in the root folder, so the
path is:
 
 C:\PySALsvn

Right-click on the folder with the mouse and then click on 'SVN checkout'.
The 'Checkout directory should be filled with the path to your folder
(`C:\PySALsvn` in this case). Copy and paste on the 'URL of repository'
space the following link:

 http://pysal.googlecode.com/svn/trunk/ pysal-read-only

Note: mind there must be a space between 'trunk/' and 'pysal-read-only'.

Once you click 'OK', a folder called 'pysal-read-only' will be created under
`C:\PySALsvn` and  all the code will be downloaded to your computer.

Now you have to tell Python to 'look for' PySAL in that folder whenever you
import it in a Python session. There are several ways to do this, here we
will use a very simple one that only implies creating a simple text file.
Open a text editor and create a file called `sitecustomize.py` located in the
Site Packages folder of you Python distribution, so the path looks more or
less like this one:
 
 C:\PythonXX\Lib\site-packages\sitecustomize.py

where XX corresponds to the version of the Python distribution you are using
(25 for 2.5, for example).

Add to the file the following text:

 import sys
 sys.path.append("C:/PySALsvn/pysal-read-only")
 
Save and close the window.

You are all set!!! Now you should be able to do the following on a Python
interactive session (on IDLE, for instance):

>>> import pysal
>>> pysal.open.check()
PySAL File I/O understands the following file extensions:
Ext: '.shp', Modes: ['r', 'wb', 'w', 'rb']
Ext: '.shx', Modes: ['r', 'wb', 'w', 'rb']
Ext: '.geoda_txt', Modes: ['r']
Ext: '.dbf', Modes: ['r', 'w']
Ext: '.gwt', Modes: ['r']
Ext: '.gal', Modes: ['r', 'w']
Ext: '.csv', Modes: ['r']
Ext: '.wkt', Modes: ['r']
>>>



