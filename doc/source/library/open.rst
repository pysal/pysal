:mod:`FileIO` -- PySAL FileIO: Module for reading and writing various file types in a Pythonic way
==================================================================================================

.. module:: FileIO
   :synopsys: Read and Write files of various types.
.. moduleauther:: Charles Schmidt <charles.r.schmidt@asu.edu>

.. versionadded:: 1.0

The PySAL FileIO system abstracts the details of file reading and writing.  All file handlers are created with the :func:`pysal.open`.


Module Contents
---------------
The :mod:`FileIO` module define the following public function:

.. function:: open(connectionString, mode='r')
   
    Parses the connectionString and mode to determire the correct file handler.
    If a custom handler is not found a python file object will be returned.
    
