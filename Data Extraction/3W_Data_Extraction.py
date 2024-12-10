# Setting Up Conda Environment

## 1. Clone the Repository
'''
Clone the repository to your local environment.
Change the directory to the the same directory where this repository is saved.
In Terminal: git clone https://github.com/petrobras/3W.git
In Terminal: cd 3W
'''

## 2. Install the Required Dependencies
'''
The toolkit dependencies listed in `environment.yml'
Create and activate the environment using Conda
In Terminal: conda env create -f environment.yml
In Terminal: conda activate 3w_env
'''
## 3. Configure path to the 3W Toolkit
'''
# Replace with full path to toolkit subfolder in the 3W repository on your local machine
In Terminal: export PYTHONPATH=$PYTHONPATH:'/path_to/3W/toolkit'
'''
## 4. Change the file path in the below code to the path specified in the comment

## 5. Run the python file in the terminal

# Imports & Configurations
import os
import pandas as pd
os.chdir(os.path.join(os.getcwd(), '..', '..'))
os.chdir('3W')
import toolkit as tk

real_instances, simulated_instances, drawn_instances = tk.get_all_labels_and_files() # Load Datasets

os.chdir(os.path.join(os.getcwd(), '..'))
os.chdir('CapStone')

# create a new folder 
if not os.path.exists('Data'):
    os.makedirs('Data')

real_data = pd.DataFrame()
for i in range(len(real_instances)): # Combining all the real instances into a single dataframe
    df = tk.load_instance(real_instances[i])
    df['Instance'] = i # Creating a new column 'Instance' to keep track of the instance number
    df['DataType'] = 'Real' # Creating a new column 'DataType' to keep track of the data source
    real_data = pd.concat([real_data, df])

real_data.to_csv('real_instances.csv', index=True) # export the real data to a csv file

simulated_data = pd.DataFrame()
for i in range(len(simulated_instances)): # Combining all the simulated instances into a single dataframe
    df = tk.load_instance(simulated_instances[i])
    df['Instance'] = i + len(real_instances) # Creating a new column 'Instance' to keep track of the instance number
    df['DataType'] = 'Simulated' # Creating a new column 'DataType' to keep track of the data source
    simulated_data = pd.concat([simulated_data, df])

simulated_data.to_csv('simulated_instances.csv', index=True) # export the simulated data to a csv file

drawn_data = pd.DataFrame()
for i in range(len(drawn_instances)): # Combining all the hand-drawn instances into a single dataframe
    df = tk.load_instance(drawn_instances[i]) 
    df['Instance'] = i + len(real_instances) + len(simulated_instances) # Creating a new column 'Instance' to keep track of the instance number
    df['DataType'] = 'Drawn' # Creating a new column 'DataType' to keep track of the data source
    drawn_data = pd.concat([drawn_data, df])

drawn_data.to_csv('hand_drawn_instances.csv', index=True) # export the drawn data to a csv file