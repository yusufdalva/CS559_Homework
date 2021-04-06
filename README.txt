Yusuf Dalva
Bilkent ID: 21602867

Said Fahri Altındiş
Bilkent ID: 22003994

----------------------
CS 559 - Homework Assignment

This assignment is prepared as a result of teamwork and has several components in it. The explanation of these components are as follows.

- data_utils.py (located in data folder): The utilities for downloading, unzipping and organizing the dataset are given in this file. This file also includes the dataset compression utilities. The dataset file is included in the Google Drive link attached to the submission.
- model.py: The implementation of the model building class, the implemented class is an instance of tensorflow.keras.Model
- Understanding the data.ipynb: Involves data I/O analysis and inspection about the data statistics
- Optimizing the model.ipynb: Inlcudes the initial optimization procedure for finding the optimal model architecture proposed in the report
- Best Model Candidate 1.ipynb: Includes the hyper parameter optimization stage for the model architecture proposed in Optimizing the model.ipynb. Final training together with final prediction analysis is given in this notebook.
- Model_Candidate_2.py: The implementation for the second model alternative (batch normalization before pooling) and the optimization process.

In our setup, we put the zip file for the dataset to the data folder.

Files submitted in the Google Drive link:

- cut_model folder: saved version of the best model trained for 600 epochs
- ckpt_model: Model weights saved for the best validation loss value
- outputs: HTML version of the outputs for the Python Notebooks submitted

For environment setup with conda:

Create conda environment using environment file in ./conda

-run 'conda env create -f environment.yml'
-Activate conda environment by 'conda activate cs559-hw'
-To have gpu support run 'conda install tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0' # For Tensorflow GPU supported build in conda