#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json


# In[12]:


from release_info import get_github_info


# In[13]:


releases = get_github_info()  


# In[15]:


releases.keys()


# In[5]:


import json
requirements = []
frozen_dict = {}
for package in releases:
    version = releases[package]['version']
    version = version.replace("v","")
    version = version.replace("V","")
    requirements.append(f'{package}>={version}')
    frozen_dict[package] = version

with open('frozen.txt', 'w') as f:
    f.write("\n".join(requirements))



with open ('../requirements.txt', 'w') as f:
    s = "\n".join(requirements)
    f.write(s)
    


# In[6]:


# update frozen.py    
with open('../pysal/frozen.py', 'w') as f:
    s = json.dumps(frozen_dict)
    s = s.replace(",", ",\n\t")
    s = f"frozen_packages = {s}"
    f.write(s)
    


# In[ ]:





# In[7]:


import pickle 
pickle.dump(releases, open( "releases.p", "wb" ) )

