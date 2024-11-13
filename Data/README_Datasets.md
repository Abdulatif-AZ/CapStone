# To extract the data directly from 3W Repository: 

1. Clone the Repository to your local environment.
Change the directory to the desired location in your local machine.
In Terminal: git clone https://github.com/petrobras/3W.git
In Terminal: cd 3W

2. Install the Required Dependencies
In Terminal: conda env create -f environment.yml
In Terminal: conda activate 3w_env

3. Configure path to the 3W Toolkit
Replace with full path to toolkit subfolder in the 3W repository on your local machine
In Terminal: export PYTHONPATH=$PYTHONPATH:'/path_to/3W/toolkit'

4. Change the file path in the below code to the path specified in the comment

5. Run the python file '3W_Data_Extraction.py' in the terminal or the through the notebook '3W_Data_Extraction.ipynb'


# To download extracted datasets:

1. Go to the Google Shared Drive for the Capstone Project

2. Navigate to 'Resourcse/3W Datasets'

3. Download zip file '3W_Datasets'

4. Extract the 3 datasets from the zip file 

5. Save the datasets in the directory where the 'CapStone' repository is located inside the 'Data' subfolder