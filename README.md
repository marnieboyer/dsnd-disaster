# Disaster Response Pipeline Project

Project Summary:
In this project, I designed a model that puts messages into categories. This is useful for disaster situations in order to route the messages in real-time to get the resources people need. The messages can be direct or through social media or news.


How to Run:
First run the ETL pipeline that cleans and stores the data in a database
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Then, fun the ML pipeline that trains the classifier and saves
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Then you run the app
python run.py

In terminal, type: env|grep WORK
You will get the spaceID and spaceDOMAIN

then naviate to the website:
https://SPACEID-3001.SPACEDOMAIN
replace the SPACEID and SPACEDOMAIN with what you got above.

Files:
data folder:
-- process_data.py 
LOADS and CLEANS the data for processing into the model

-- disaster_categories.csv -Part of the training data, categories for the messages

-- disaster_messages.csv -Part of the training data, messages

-- DisasterResponse.db - the database that will be created in step 1

models
-- train_classifier.py -- this creates, fits and saves the classifier model for assigning the messages to categories

-- classifier.pkl -- the saved model

app
--run.py -- this file pulls it all together into the app with visualizations
