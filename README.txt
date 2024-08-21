5/8/2022
Team Chipmunk

1. Install Anaconda and create a virtual environment with Python 3.8.5

Create the virtual environment in Anaconda prompt by entering:
	
	conda create -n myenv python=3.8.5

Note: myenv is an example name

Activate the virtual environment by entering:

	conda activate myenv

2. Install required libraries with the command

   	pip install -r requirements.txt

If user permission is required to install, you may try the following instead:

	pip install -r requirements.txt --user


3. Extract data/RAVDESS.zip to data/RAVDESS

4. Select the emotions to use from RAVDESS 
   and prepare training data with create_training_data.ipynb

The preprocessed data will be updated in data/preprocessed
   
5. Train a model with train_model.ipynb 

The model will be saved in the folder models

6. Run the application from Anaconda Prompt with the command

   python streaming.py