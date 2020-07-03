#!/usr/bin/env python
# coding: utf-8

# In[1]:


from phue import Bridge
import time
import pickle


# In[2]:


b = Bridge('192.168.178.20')
b.connect()
b.get_api()
b.set_light(2,'on', True)


# In[ ]:


while True:
    msg = pickle.load(open('./lightMSG.pk', 'rb'))

    # Set brightness of lamp 1 based on mood message
    if msg == 'happy': ##purple
        b.set_light(2, 'hue', 50000)
        b.set_light(2, 'bri', 250)
        b.set_light(2, 'sat', 250)
    if msg == 'sad': ##blue
        b.set_light(2, 'hue', 40000)
        b.set_light(2, 'sat', 250)
        b.set_light(2, 'bri', 200)
    if msg == 'neutral': ##white
        b.set_light(2, 'hue', 40000)
        b.set_light(2, 'sat', 0)
        b.set_light(2, 'bri', 200)
    
    time.sleep(2)


# In[ ]:




