# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:12:27 2024
@author: HuLab
"""

import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the input file
input_file_name = 'result_car_truck_xception.txt'

# Read the file into a DataFrame
data = pd.read_csv(input_file_name, sep=';', header=None, usecols=[1, 5, 6],
                   names=['Class', 'Pred-gt', 'Absolute Relative Error'])

# Group the data by 'Class' and calculate the mean and variance of the Pred-gts
grouped_data = data.groupby('Class').agg({'Pred-gt': ['mean', 'var'], 'Absolute Relative Error': ['mean', 'var']})

# Calculate overall mean and variance for Pred-gt and Absolute Relative Error
overall_stats = data.agg({'Pred-gt': ['mean', 'var'], 'Absolute Relative Error': ['mean', 'var']})

# Print the mean and variance of Pred-gt and Absolute Relative Error for each class
print("Mean and Variance for each class:")
print(grouped_data)

# Print the overall mean and variance
print("\nOverall Mean and Variance:")
print(overall_stats)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Adjust the subplot grid

# Mean Pred-gt plot for each class
grouped_data['Pred-gt']['mean'].plot(kind='bar', color='skyblue', ax=axs[0, 0])
axs[0, 0].set_title('Mean Pred-gt by Class')
axs[0, 0].set_ylabel('Mean Pred-gt')
axs[0, 0].set_xlabel('Class')

# Mean Absolute Relative Error plot for each class
(grouped_data['Absolute Relative Error']['mean']*100).plot(kind='bar', color='salmon', ax=axs[1, 0])
axs[1, 0].set_title('Mean Absolute Relative Error by Class')
axs[1, 0].set_ylabel('Mean Absolute Relative Error')
axs[1, 0].set_xlabel('Class')

# Variance Pred-gt plot for each class
grouped_data['Pred-gt']['var'].plot(kind='bar', color='skyblue', ax=axs[0, 1])
axs[0, 1].set_title('Variance of Pred-gt by Class')
axs[0, 1].set_ylabel('Variance Pred-gt')
axs[0, 1].set_xlabel('Class')

# Variance Absolute Relative Error plot for each class
(grouped_data['Absolute Relative Error']['var'] * 100).plot(kind='bar', color='salmon', ax=axs[1, 1])
axs[1, 1].set_title('Variance of Absolute Relative Error by Class')
axs[1, 1].set_ylabel('Variance Absolute Relative Error')
axs[1, 1].set_xlabel('Class')

# Overall Mean Pred-gt and Absolute Relative Error
axs[0, 2].bar(['Pred-gt', 'Absolute Relative Error'], overall_stats.loc['mean'], color=['skyblue', 'salmon'])
axs[0, 2].set_title('Overall Mean Pred-gt and Absolute Relative Error')
axs[0, 2].set_ylabel('Value')

# Overall Variance Pred-gt and Absolute Relative Error
axs[1, 2].bar(['Pred-gt', 'Absolute Relative Error'], overall_stats.loc['var']*100, color=['skyblue', 'salmon'])
axs[1, 2].set_title('Overall Variance Pred-gt and Absolute Relative Error')
axs[1, 2].set_ylabel('Value')

# Improve layout and show the plot
plt.tight_layout()
plt.show()
