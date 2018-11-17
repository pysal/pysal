Thank you for filing this issue! To help troubleshoot this issue, please follow
the following directions to the best of your ability before submitting an issue.
Feel free to delete this text once you've filled out the relevant requests. 

Please include the output of the following in your issue submission. If you don't know how to provide the information, commands to get the relevant information from the Python interpreter will follow each bullet point.

Feel free to delete the commands after you've filled out each bullet. 

- Platform information:
```python
>>> import os; print(os.name, os.sys.platform);print(os.uname())
```
- Python version: 
```python
>>> import sys; print(sys.version)
```
- SciPy version:
```python
>>> import scipy; print(scipy.__version__)
```
- NumPy version:
```python
>>> import numpy; print(numpy.__version__)
```

Also, please upload any relevant data as [a file
attachment](https://help.github.com/articles/file-attachments-on-issues-and-pull-requests/). Please **do not** upload pickled objects, since it's nearly impossible to troubleshoot them without replicating your exact namespace. Instead, provide the minimal subset of the data required to replicate the problem. If it makes you more comfortable submitting the issue, feel free to:

1. remove personally identifying information from data or code
2. provide only the required subset of the full data or code 
