import  pandas as pd
import os
import re
import numpy as np
path = os.getcwd() + '\izeqi\data\data_new_1.txt'
data = pd.read_csv(path, header=None,delim_whitespace=True, names=('plotRatio',
                                             'transactionDate',
                                             'floorPrice',
                                            ))
b=[]
Date1=data.values[:,1]
i = 0
f= open(path,'r',encoding='utf-8')
words = f.read().split()
for word in words:
     a=re.match(r'(.*)-(.*)-(.*)', word, re.M | re.I)
     if a:
        year=int(a.group(1))
        month=int(a.group(2))
        day=int(a.group(3))
        date=year+(month-1)/12+(day-1)/365
        a=re.sub(r'(.*)-(.*)-(.*)',str(date), word, re.M | re.I)
        b.append(a)
m=0
n=0
for i in range(len(b)):
    Date1[m]=b[i]
    m=m+1


data['transactionDate']=Date1
data.to_csv("data_new_1.txt")