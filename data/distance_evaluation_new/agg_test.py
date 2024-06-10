# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 03:27:20 2024

@author: yukai
"""

import pandas as pd

data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

# Calculate mean and standard deviation for both columns
overall_stats = data.agg({'col1': ['mean', 'std'], 'col2': ['mean', 'std']})

print(overall_stats)
