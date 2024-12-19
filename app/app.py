from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import pickle
import os
from utils import preprocess_data  # Import the function

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models/model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '../data/Mental_Health_Care_in_the_Last_4_Weeks.csv')

# Load model
with open(MODEL_PATH, 'rb') as model_file:
    trained_model = pickle.load(model_file)

# Dummy columns
dummy_columns = ['Group_Subgroup_By Age_18 - 29 years',
 'Group_Subgroup_By Age_30 - 39 years',
 'Group_Subgroup_By Age_40 - 49 years',
 'Group_Subgroup_By Age_50 - 59 years',
 'Group_Subgroup_By Age_60 - 69 years',
 'Group_Subgroup_By Age_70 - 79 years',
 'Group_Subgroup_By Age_80 years and above',
 'Group_Subgroup_By Disability status_With disability',
 'Group_Subgroup_By Disability status_Without disability',
 "Group_Subgroup_By Education_Bachelor's degree or higher",
 'Group_Subgroup_By Education_High school diploma or GED',
 'Group_Subgroup_By Education_Less than a high school diploma',
 "Group_Subgroup_By Education_Some college/Associate's degree",
 'Group_Subgroup_By Gender identity_Cis-gender female',
 'Group_Subgroup_By Gender identity_Cis-gender male',
 'Group_Subgroup_By Gender identity_Transgender',
 'Group_Subgroup_By Presence of Symptoms of Anxiety/Depression_Did not experience symptoms of anxiety/depression in the past 4 weeks',
 'Group_Subgroup_By Presence of Symptoms of Anxiety/Depression_Experienced symptoms of anxiety/depression in past 4 weeks',
 'Group_Subgroup_By Race/Hispanic ethnicity_Hispanic or Latino',
 'Group_Subgroup_By Race/Hispanic ethnicity_Non-Hispanic Asian, single race',
 'Group_Subgroup_By Race/Hispanic ethnicity_Non-Hispanic Black, single race',
 'Group_Subgroup_By Race/Hispanic ethnicity_Non-Hispanic White, single race',
 'Group_Subgroup_By Race/Hispanic ethnicity_Non-Hispanic, other races and multiple races',
 'Group_Subgroup_By Sex_Female',
 'Group_Subgroup_By Sex_Male',
 'Group_Subgroup_By Sexual orientation_Bisexual',
 'Group_Subgroup_By Sexual orientation_Gay or lesbian',
 'Group_Subgroup_By Sexual orientation_Straight',
 'Group_Subgroup_By State_Alabama',
 'Group_Subgroup_By State_Alaska',
 'Group_Subgroup_By State_Arizona',
 'Group_Subgroup_By State_Arkansas',
 'Group_Subgroup_By State_California',
 'Group_Subgroup_By State_Colorado',
 'Group_Subgroup_By State_Connecticut',
 'Group_Subgroup_By State_Delaware',
 'Group_Subgroup_By State_District of Columbia',
 'Group_Subgroup_By State_Florida',
 'Group_Subgroup_By State_Georgia',
 'Group_Subgroup_By State_Hawaii',
 'Group_Subgroup_By State_Idaho',
 'Group_Subgroup_By State_Illinois',
 'Group_Subgroup_By State_Indiana',
 'Group_Subgroup_By State_Iowa',
 'Group_Subgroup_By State_Kansas',
 'Group_Subgroup_By State_Kentucky',
 'Group_Subgroup_By State_Louisiana',
 'Group_Subgroup_By State_Maine',
 'Group_Subgroup_By State_Maryland',
 'Group_Subgroup_By State_Massachusetts',
 'Group_Subgroup_By State_Michigan',
 'Group_Subgroup_By State_Minnesota',
 'Group_Subgroup_By State_Mississippi',
 'Group_Subgroup_By State_Missouri',
 'Group_Subgroup_By State_Montana',
 'Group_Subgroup_By State_Nebraska',
 'Group_Subgroup_By State_Nevada',
 'Group_Subgroup_By State_New Hampshire',
 'Group_Subgroup_By State_New Jersey',
 'Group_Subgroup_By State_New Mexico',
 'Group_Subgroup_By State_New York',
 'Group_Subgroup_By State_North Carolina',
 'Group_Subgroup_By State_North Dakota',
 'Group_Subgroup_By State_Ohio',
 'Group_Subgroup_By State_Oklahoma',
 'Group_Subgroup_By State_Oregon',
 'Group_Subgroup_By State_Pennsylvania',
 'Group_Subgroup_By State_Rhode Island',
 'Group_Subgroup_By State_South Carolina',
 'Group_Subgroup_By State_South Dakota',
 'Group_Subgroup_By State_Tennessee',
 'Group_Subgroup_By State_Texas',
 'Group_Subgroup_By State_Utah',
 'Group_Subgroup_By State_Vermont',
 'Group_Subgroup_By State_Virginia',
 'Group_Subgroup_By State_Washington',
 'Group_Subgroup_By State_West Virginia',
 'Group_Subgroup_By State_Wisconsin',
 'Group_Subgroup_By State_Wyoming']

# Helper function to transform user input into dummy variables
def transform_input_to_dummy(user_input, value):
    """
    Convert user input into dummy variables for the model and scale by 'Value'.
    """
    dummy_input = pd.DataFrame(0, index=[0], columns=dummy_columns)
    for key, val in user_input.items():
        column_name = f"Group_Subgroup_By {key}_{val}"
        if column_name in dummy_columns:
            dummy_input.at[0, column_name] = 1

    # Scale the dummy variables by 'Value'
    for col in dummy_input.columns:
        if col.startswith('Group_Subgroup_By'):
            dummy_input[col] *= value

    return dummy_input
def create_composite_dummy(user_input, data):
    """
    Create a composite dummy input for a single user based on matched subgroups.
    """
    matching_rows = pd.DataFrame()
    
    # Match rows based on each input feature
    for key, value in user_input.items():
        if key == "Name":  # Skip Name field
            continue
        column_name = f"Group_Subgroup_By {key}_{value}"
        if column_name in data.columns:
            matches = data[data[column_name] == 1]
            matching_rows = pd.concat([matching_rows, matches], ignore_index=True)

    # Drop duplicates
    matching_rows = matching_rows.drop_duplicates()

    # Aggregate relevant rows
    if matching_rows.empty:
        return None

    aggregated_value = matching_rows['Value'].sum()
    composite_dummy = transform_input_to_dummy(user_input, aggregated_value)

    return composite_dummy

@app.route('/')
def login():
    return render_template('login.html')

@app.post('/login')
def validate_login():
    """
    Validate the user's login details and preprocess input to match dummy columns.
    """
    # Preprocess the dataset
    data = preprocess_data(DATA_PATH)

    # Collect user input from the login form
    user_input = {
        "Age": request.form.get('age'),
        "Sex": request.form.get('sex'),
        "Disability status": request.form.get('disability_status'),
        "Education": request.form.get('education'),
        "Gender identity": request.form.get('gender_identity'),
        "Symptoms": request.form.get('symptoms'),
        "Race": request.form.get('race'),  # Simplify key
        "Sexual Orientation": request.form.get('sexual_orientation'),
        "State": request.form.get('state'),
        "Name": request.form.get('name')
    }

    print("User Input:", user_input)
    print("Dataset Columns:", data.columns)

    # Initialize dummy input with zeroes
    dummy_input = pd.DataFrame(0, index=[0], columns=dummy_columns)

    # Iterate over each feature in user input and scale matched features
    for key, value in user_input.items():
        if key == "Name":  # Skip name since it doesn't correspond to a dataset column
            continue
        column_name = f"Group_Subgroup_By {key}_{value}"
        if column_name in data.columns:
            # Find matching rows for this feature
            matching_rows = data[data[column_name] == 1]
            if not matching_rows.empty:
                # Scale the dummy variable by the corresponding 'Value'
                dummy_input[column_name] = matching_rows['Value'].median()
                print(f"Scaled {column_name} with median value {matching_rows['Value'].median()}")
        else:
            print(f"Column {column_name} does not exist in the dataset.")

    # If the dummy input remains all zeros, it means no match was found
    if dummy_input.sum().sum() == 0:
        flash("No matching demographic data found. Please check your inputs.", "error")
        return redirect(url_for('login'))

    # Debug: Print the final dummy input
    print("Final Dummy Input for Model:")
    print(dummy_input)

    # Store user details and dummy input in session
    session['patient'] = user_input
    session['dummy_input'] = dummy_input.to_dict(orient='records')[0]

    flash('Login successful!', 'success')
    return redirect(url_for('home'))


@app.route('/assessment', methods=['GET', 'POST'])
def assessment():
    if request.method == 'POST':
        # Debug: Print form responses
        responses = request.form.to_dict()
        print("Form Responses:", responses)

        # Initialize section scores
        scores = {}

        # Calculate scores
        for question, value in responses.items():
            section = question.split('_')[0]  # Extract the section from the question name
            if section not in scores:
                scores[section] = []
            scores[section].append(int(value))

        # Calculate averages and total score
        section_averages = {section: np.mean(values) for section, values in scores.items()}
        total_score = np.sum(list(section_averages.values()))

        return render_template('result.html', total_score=total_score, section_averages=section_averages)

    return render_template('assessment.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/home')
def home():
    patient = session.get('patient')
    if not patient:
        flash('Please log in to continue.', 'error')
        return redirect(url_for('login'))
    return render_template('home.html', patient=patient)

@app.route('/profile')
def profile():
    patient = session.get('patient')
    print("Session Patient Data:", patient)  # Debugging
    subscription_plan = session.get('subscription_plan')
    return render_template('profile.html', patient=patient, subscription_plan=subscription_plan)

@app.route('/generate-quote', methods=['GET', 'POST'])
def generate_quote():
    """
    Generate a subscription plan based on the patient's data, preserving feature structure.
    """
    if request.method == 'POST':
        print("Generate Quote Route Hit")

        # Retrieve the dummy input from the session
        dummy_input_data = session.get('dummy_input')
        if not dummy_input_data:
            flash("No data found for generating a quote. Please log in again.", "error")
            return redirect(url_for('login'))

        # Convert dummy input back to a DataFrame for prediction
        dummy_input = pd.DataFrame([dummy_input_data])

        # Debug: Print dummy input
        print("Dummy Input for Prediction:")
        print(dummy_input)

        # Predict using the trained model
        try:
            predictions = trained_model.predict(dummy_input)

            # Debug: Print predictions
            print("Predictions for Dummy Input:")
            print(predictions)

            # Aggregate predictions if needed (e.g., majority vote for ensemble models)
            from collections import Counter
            aggregated_prediction = Counter(predictions).most_common(1)[0][0]

            # Debug: Print aggregated prediction
            print("Aggregated Prediction:")
            print(aggregated_prediction)
            # Evaluate the model
            aggregated_prediction = aggregated_prediction.replace(", Last 4 Weeks", "")

            # Map prediction to subscription plans
            plan_mapping = {
                "Needed Counseling or Therapy But Did Not Get It": "Free",
                "Received Counseling or Therapy": "Basic",
                "Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy": "Premium",
                "Took Prescription Medication for Mental Health": "Advanced"
            }

            # Determine the recommended plan
            subscription_plan = plan_mapping.get(aggregated_prediction, "Unknown Plan")

            # Store the subscription plan in the session
            session['subscription_plan'] = subscription_plan

            # Render the result on the generate_quote page
            return render_template('generate_quote.html', subscription_plan=subscription_plan)

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash("An error occurred during the prediction process. Please try again.", "error")
            return redirect(url_for('home'))

    # For GET requests, render the page with no subscription plan yet
    return render_template('generate_quote.html', subscription_plan=None)

if __name__ == '__main__':
    app.run(debug=True)