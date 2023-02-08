# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:02:28 2022

@author: valero
"""
import numpy as np
a=[-25,25]
b=[25,25]

norm_a=np.linalg.norm(a)
norm_b=np.linalg.norm(b)

unit_a=a/norm_a
unit_b=b/norm_b


angle=np.arccos(sum(unit_a*unit_b))