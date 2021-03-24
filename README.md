# Disaster Response Pipeline Project
### Project Motivation
This project was completed as part of my Udacity Data Science Nano Degree. The project aims to help aid workers during disaster response. It helps to map messages or events to appropriate disaster relief agency. The tool is intended to replace regular key word searches with a more accurate method of classifying messages.
The project includes a web app where an emergency worker can type a message and get classification results in multpiple categories.

Below is a screenshot of the webapp

[my image] (Webapp.png)



### Project Components
1. ETL Pipeline

In process_data.py

	- Load the messages and categories datasets
    - Merges the two datasets
	- Cleans the data
	- Stores it in a SQLite database

2. ML Pipeline
In train_classifier.py
	Load data from SQLite database
    Split the dataset into training and test sets
    Build text processing and machine learning pipeline
    Train and tune model using GridSearchCV
    Output results
    Export final model as a pickle file
    
3. Flask Web App
Much of these was already provided
	Modified file paths for database and model as needed




### Instructions:
Here are the instructions on how to run the web app
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgement
Thanks to Figure Eight for providing data used on this project. Many thanks to the Udacity Data Science course team for teaching and providing supporting materials without which this project would not be possible. Thanks to the many Python developers that worked on Libraries used in this project.