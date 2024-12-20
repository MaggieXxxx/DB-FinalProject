# Database Systems CSCI-GA.2433 Fall 2023 - Final Project

Team Members: Maggie Xu (jx1206), Lia Wang (rw2618), Ziyu Qi (zq2127)

## Demo of Mental Health Subscription Plan Recommender
![](https://github.com/MaggieXxxx/DB-FinalProject/blob/main/Demo.gif)

## Description
This project leverages machine learning to recommend personalized subscription plans for mental health care based on demographic, personal, and historical data. Users input details such as age, gender, state, and symptoms of mental health conditions, and the system predicts the most suitable plan using a trained model. The application is built with Flask, and the predictions are powered by a Random Forest model trained on a subgroup-based dataset.

### Features
	•	User Login: Input demographic and personal information through a user-friendly form.
	•	Recommendation Engine: Predicts the best subscription plan based on input data using a trained machine learning model.
	•	Profile Page: Displays user details and the recommended subscription plan.
	•	Dynamic UI: Modern web design with responsive components for a seamless user experience.
	•	Data Handling: Preprocessed and normalized user data to align with the subgroup dataset for accurate predictions.

### Tech Stack
	•	Backend: Flask
	•	Frontend: HTML, CSS, Jinja2 Templates
	•	Machine Learning: Scikit-learn (Random Forest Classifier)
	•	Data: CSV dataset of mental health care demographics
	•	Environment: Python 3.9+

### How to Run
#### 1.	Start the Flask application:
```bash
cd path/to/app
```
```bash
python app.py
```

#### 2.	Open your browser and navigate to:
```bash
http://127.0.0.1:5000/
```

#### 3.	Use the login form to enter demographic details and generate a personalized subscription plan.

### Project Structure
```plaintext
├── README.md                        # Project description and instructions
├── app                              # Main application folder
│   ├── __pycache__                  # Compiled Python cache files
│   │   ├── calculateQuote.cpython-311.pyc
│   │   └── utils.cpython-312.pyc
│   ├── app.py                       # Main Flask application script
│   ├── models                       # Trained model and encoder files
│   │   ├── encoder.pkl              # Preprocessing encoder
│   │   └── model.pkl                # Trained machine learning model
│   ├── static                       # Static assets
│   │   ├── css                      # Stylesheets
│   │   │   └── styles.css
│   │   └── js                       # JavaScript files
│   │       └── main.js
│   ├── templates                    # HTML templates
│   │   ├── generate_quote.html      # Subscription recommendation page
│   │   ├── home.html                # Home page
│   │   ├── index.html               # Base template
│   │   ├── login.html               # Login page
│   │   └── profile.html             # User profile page
│   └── utils.py                     # Utility functions for data preprocessing
└── data                             # Dataset folder
    └── Mental_Health_Care_in_the_Last_4_Weeks.csv
                                      # Demographic data for training and predictions
```

### Dataset
* The dataset (Mental_Health_Care_in_the_Last_4_Weeks.csv) contains demographic subgroup information and mental health care outcomes. Key fields include:\
	•	Group/Subgroup Features: Age, Sex, Disability Status, etc.\
	•	Indicator: Outcomes related to mental health care.\
	•	Value: Metrics used for scaling in the prediction process.
