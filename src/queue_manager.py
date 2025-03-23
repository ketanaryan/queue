import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_cors import CORS
import os
import logging
import random  # For random early notification time

# Ensure the working directory is the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Setup logging for audit trails
logging.basicConfig(
    filename='queue_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the trained model pipeline
model_pipeline = joblib.load('../models/wait_time_model.pkl')

# Load synthetic data for background dataset (needed for KernelExplainer)
synthetic_data = pd.read_csv('../data/synthetic_data.csv')

# Define features (raw features before preprocessing)
raw_features = [
    'hour', 'day_of_week', 'queue_length', 'is_emergency', 'priority_score',
    'doctors_available', 'patient_condition_severity', 'delay_factor', 'is_canceled',
    'doctor_experience', 'patient_age', 'historical_wait_time', 'queue_position',
    'is_peak_hour', 'is_night_shift', 'is_specialized',
    'patients_per_doctor', 'queue_efficiency', 'is_senior', 'is_child',
    'emergency_severity', 'critical_case', 'is_weekend', 'department', 'time_of_day',
    'is_follow_up'
]

# Define numerical and categorical features for preprocessing
numerical_features = [
    'hour', 'day_of_week', 'queue_length', 'is_emergency', 'priority_score',
    'doctors_available', 'patient_condition_severity', 'delay_factor', 'is_canceled',
    'doctor_experience', 'patient_age', 'historical_wait_time', 'queue_position',
    'is_peak_hour', 'is_night_shift', 'is_specialized',
    'patients_per_doctor', 'queue_efficiency', 'is_senior', 'is_child',
    'emergency_severity', 'critical_case', 'is_weekend',
    'is_follow_up'
]
categorical_features = ['department', 'time_of_day']

# Get the transformed feature names after preprocessing
preprocessor = model_pipeline.named_steps['preprocessor']
transformed_feature_names = (
    numerical_features +
    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
)

# Map technical feature names to user-friendly names for SHAP explanations
feature_name_mapping = {
    'doctor_experience': 'Doctor Experience',
    'is_canceled': 'Recent Cancellations',
    'queue_efficiency': 'Queue Progress',
    'priority_score': 'Priority Score',
    'patient_age': 'Patient Demographics',
    'is_senior': 'Patient Demographics',
    'is_child': 'Patient Demographics',
    'patients_per_doctor': 'Staff Workload',
    'queue_length': 'Queue Size',
    'doctors_available': 'Available Doctors',
    'patient_condition_severity': 'Condition Severity',
    'emergency_severity': 'Emergency Severity',
    'critical_case': 'Critical Case',
    'is_emergency': 'Emergency Status',
    'is_specialized': 'Specialized Care',
    'is_peak_hour': 'Peak Hour',
    'is_night_shift': 'Night Shift',
    'is_weekend': 'Weekend',
    'hour': 'Time of Day',
    'day_of_week': 'Day of Week',
    'queue_position': 'Queue Position',
    'historical_wait_time': 'Historical Wait Time',
    'delay_factor': 'Delays',
    'is_follow_up': 'Follow-Up Status'
}

# Define required raw columns
required_raw_columns = [
    'department', 'patient_condition_severity', 'day_of_week', 'hour',
    'queue_length', 'doctors_available', 'queue_position', 'patient_age',
    'delay_factor', 'is_canceled', 'doctor_experience', 'historical_wait_time',
    'is_follow_up'
]

def engineer_features(df):
    """Centralized feature engineering function"""
    df = df.copy()
    
    # Ensure required raw columns are present
    for col in required_raw_columns:
        if col not in df:
            df[col] = 0

    # Feature engineering
    df['is_emergency'] = (df['department'] == 'Emergency').astype(int)
    df['is_specialized'] = df['department'].apply(
        lambda x: 1 if x in ['Cardiology', 'Orthopedics', 'Pediatrics'] else 0
    )
    df['priority_score'] = df['patient_condition_severity'] * 0.8
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night_shift'] = df['hour'].apply(lambda x: 1 if x < 8 or x >= 20 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (14 <= x <= 17) else 0)
    df['time_of_day'] = df['hour'].apply(lambda x: 
        'morning' if 5 <= x < 12 else
        'afternoon' if 12 <= x < 17 else
        'evening' if 17 <= x < 22 else
        'night'
    )
    df['patients_per_doctor'] = df['queue_length'] / np.maximum(df['doctors_available'], 1)
    df['queue_efficiency'] = (df['queue_length'] - df['queue_position']) / np.maximum(df['queue_length'], 1)
    df['is_senior'] = (df['patient_age'] >= 65).astype(int)
    df['is_child'] = (df['patient_age'] <= 12).astype(int)
    df['emergency_severity'] = df['is_emergency'] * df['patient_condition_severity']
    df['critical_case'] = (df['patient_condition_severity'] >= 3).astype(int)

    # Ensure all features are present
    for feature in raw_features:
        if feature not in df.columns:
            df[feature] = 0

    return df

# Apply feature engineering to synthetic data
synthetic_data = engineer_features(synthetic_data)

# Initialize SHAP explainer
try:
    if 'model' not in model_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")
    xgb_model = model_pipeline.named_steps['model']
    if not isinstance(xgb_model, xgb.XGBRegressor):
        raise ValueError(f"Expected xgb.XGBRegressor, but got {type(xgb_model)}")
    explainer = shap.TreeExplainer(xgb_model)
    print("Using TreeExplainer for SHAP explanations.")
except Exception as e:
    print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
    background_data = synthetic_data[raw_features].head(100)
    explainer = shap.KernelExplainer(model_pipeline.predict, background_data)

# Initialize queue and feedback data
queue = {}
feedback_data = []

def predict_base_wait_time(patient_data):
    """Predict the base wait time for a patient (time if they were at the front of the queue)"""
    patient_df = pd.DataFrame([patient_data])
    patient_df = engineer_features(patient_df)
    
    X = patient_df[raw_features]
    base_wait_time = model_pipeline.predict(X)[0]
    
    # Prioritize Emergency department patients
    if patient_data.get('department') == 'Emergency':
        base_wait_time *= 0.5  # Reduce wait time by 50% for Emergency patients
    
    # Prioritize follow-up patients
    if patient_data.get('is_follow_up', 0) == 1:
        base_wait_time *= 0.7  # Reduce wait time by 30% for follow-up patients
        print(f"Adjusted base wait time for follow-up patient {patient_data['patient_id']} by 30% to {base_wait_time:.1f} minutes.")
    
    # Ensure non-negative wait time and apply fairness constraint
    base_wait_time = max(1, base_wait_time)
    if not patient_data.get('is_emergency', 0) and base_wait_time > 60:
        base_wait_time = 60
        print(f"Adjusted base wait time for regular patient {patient_data['patient_id']} to 60 minutes for fairness.")
    
    return float(base_wait_time)  # Convert to Python float

def shap_explanation(patient_data):
    """Generate SHAP explanation for the prediction with user-friendly feature names"""
    patient_df = pd.DataFrame([patient_data])
    patient_df = engineer_features(patient_df)
    
    if isinstance(explainer, shap.TreeExplainer):
        preprocessor = model_pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(patient_df[raw_features])
        shap_values = explainer.shap_values(X_transformed)
    else:
        X = patient_df[raw_features]
        shap_values = explainer.shap_values(X)
    
    # Debug: Print lengths to identify mismatch
    logger.info(f"Length of transformed_feature_names: {len(transformed_feature_names)}")
    logger.info(f"Length of shap_values[0]: {len(shap_values[0])}")
    logger.info(f"Transformed feature names: {transformed_feature_names}")
    logger.info(f"SHAP values: {shap_values[0]}")
    
    # Map transformed feature names to user-friendly names
    shap_importance = pd.DataFrame({
        'feature': transformed_feature_names,
        'value': shap_values[0]
    })
    
    # Simplify feature names for display
    shap_importance['display_feature'] = shap_importance['feature'].apply(
        lambda x: feature_name_mapping.get(x.split('_')[0] if '_' in x else x, x)
    )
    
    # Aggregate SHAP values for one-hot encoded features (e.g., department_Emergency, department_General)
    aggregated_shap = shap_importance.groupby('display_feature')['value'].sum().reset_index()
    top_features = aggregated_shap.sort_values(by='value', key=abs, ascending=False).head(3)
    
    return ', '.join([f"{row['display_feature']} ({row['value']:.2f})" for _, row in top_features.iterrows()])

def notify_patient(patient_id, message):
    """Simulate sending a notification to a patient"""
    print(f"Notification to Patient {patient_id}: {message}")
    logger.info(f"Notification sent to Patient {patient_id}: {message}")

def recalculate_wait_times():
    """Recalculate predicted wait times for all patients based on queue position"""
    global queue
    if not queue:
        return

    # Sort patients by queue position
    sorted_patients = sorted(queue.items(), key=lambda x: x[1]['queue_position'])
    
    cumulative_time = 0.0
    for idx, (pid, patient) in enumerate(sorted_patients):
        # The predicted wait time is the cumulative time of all patients ahead plus the patient's own base wait time and cumulative delay
        patient['predicted_wait_time'] = cumulative_time + patient['base_wait_time'] + patient['cumulative_delay']
        # Ensure predicted wait time doesn't go below 1 minute
        patient['predicted_wait_time'] = max(1, patient['predicted_wait_time'])
        cumulative_time = patient['predicted_wait_time']

        # Apply early notification to ALL patients
        early_notification = random.uniform(5, 10)  # Randomly choose between 5 and 10 minutes
        notified_wait_time = max(1, patient['predicted_wait_time'] - early_notification)
        patient['notified_wait_time'] = notified_wait_time
        print(f"Patient {pid} (position {patient['queue_position']}) is being notified {early_notification:.1f} minutes earlier than predicted wait time.")
        logger.info(f"Patient {pid} (position {patient['queue_position']}) notified {early_notification:.1f} minutes early. Predicted Wait Time: {patient['predicted_wait_time']:.1f}, Notified Wait Time: {notified_wait_time:.1f}")

        # Update the queue with the new patient data
        queue[pid] = patient
        print(f"Patient {pid}: Recalculated wait time: {patient['predicted_wait_time']:.1f} minutes, Notified wait time: {patient['notified_wait_time']:.1f} minutes.")
        notify_patient(pid, f"Your estimated wait time is now {patient['notified_wait_time']:.1f} minutes after queue recalculation.")

def update_queue(patient_data):
    """Add a patient to the queue and predict their wait time"""
    global queue
    patient_id = patient_data['patient_id']
    new_position = patient_data['queue_position']
    queue_length = patient_data['queue_length']

    # Validate and adjust queue positions
    for pid, patient in queue.items():
        if patient['queue_position'] >= new_position:
            patient['queue_position'] += 1
        patient['queue_length'] = max(patient['queue_length'], queue_length)

    # Calculate priority_score explicitly
    patient_data['priority_score'] = patient_data['patient_condition_severity'] * 0.8

    base_wait_time = predict_base_wait_time(patient_data)

    # Initialize cumulative delay and notification time for the patient
    patient_data['base_wait_time'] = base_wait_time
    patient_data['cumulative_delay'] = 0
    patient_data['predicted_wait_time'] = base_wait_time
    patient_data['notified_wait_time'] = base_wait_time  # Will be updated by recalculate_wait_times

    queue[patient_id] = patient_data

    # Recalculate wait times for all patients
    recalculate_wait_times()

    print(f"\nPatient {patient_id} Details:")
    print(f"  Department: {patient_data['department']}")
    print(f"  Condition Severity: {patient_data['patient_condition_severity']}")
    print(f"  Priority Score: {patient_data['priority_score']:.1f}")
    print(f"  Follow-Up: {'Yes' if patient_data.get('is_follow_up', 0) == 1 else 'No'}")
    print(f"  Queue Position: {patient_data['queue_position']}/{patient_data['queue_length']}")
    print(f"  Base wait time: {base_wait_time:.1f} minutes.")
    print(f"  Predicted wait time: {patient_data['predicted_wait_time']:.1f} minutes.")
    print(f"  Notified wait time: {patient_data['notified_wait_time']:.1f} minutes.")
    print(f"  Influenced by: {shap_explanation(patient_data)}")

    # Notify patient of their estimated wait time
    notify_patient(patient_id, f"Your estimated wait time is {patient_data['notified_wait_time']:.1f} minutes.")

    # Log the action with more details
    logger.info(f"Patient {patient_id} added to queue: {patient_data}, Base Wait Time: {patient_data['base_wait_time']:.1f}, Cumulative Delay: {patient_data['cumulative_delay']:.1f}, Predicted Wait Time: {patient_data['predicted_wait_time']:.1f}, Notified Wait Time: {patient_data['notified_wait_time']:.1f}")

    print_queue()
    return queue

def handle_missed_appointment(patient_id, new_position_after_patient_id):
    """Handle a patient who missed their turn and arrives after another patient"""
    global queue
    if patient_id not in queue:
        print(f"Patient {patient_id} not found in queue.")
        return queue
    if new_position_after_patient_id not in queue:
        print(f"Patient {new_position_after_patient_id} not found in queue.")
        return queue

    # Get the current and new positions
    missed_patient = queue[patient_id]
    current_position = missed_patient['queue_position']
    after_patient = queue[new_position_after_patient_id]
    new_position = after_patient['queue_position'] + 1

    # Remove the missed patient temporarily
    del queue[patient_id]

    # Adjust queue positions for patients between current_position and new_position
    for pid, patient in queue.items():
        if current_position < patient['queue_position'] <= new_position:
            patient['queue_position'] -= 1
        elif patient['queue_position'] >= new_position:
            patient['queue_position'] += 1

    # Reinsert the missed patient at the new position
    missed_patient['queue_position'] = new_position
    queue[patient_id] = missed_patient

    # Update queue length for all patients
    queue_length = len(queue)
    for pid, patient in queue.items():
        patient['queue_length'] = queue_length

    print(f"Patient {patient_id} missed their turn and is now placed after Patient {new_position_after_patient_id} at position {new_position}.")

    # Notify the missed patient
    notify_patient(patient_id, f"You missed your turn. You have been placed at position {new_position} in the queue.")

    # Log the action
    logger.info(f"Patient {patient_id} missed their turn and was moved to position {new_position} after Patient {new_position_after_patient_id}.")

    # Recalculate wait times for all patients
    recalculate_wait_times()

    print_queue()
    return queue

def handle_delay(patient_id, delay_minutes):
    """Handle a delay for a patient and update the queue"""
    global queue
    if patient_id not in queue:
        print(f"Patient {patient_id} not found in queue.")
        return queue

    delayed_patient = queue[patient_id]
    delayed_position = delayed_patient['queue_position']
    
    # Update the delayed patient's cumulative delay
    delayed_patient['cumulative_delay'] += delay_minutes
    print(f"Patient {patient_id} delayed by {delay_minutes} minutes.")

    # Notify the delayed patient
    notify_patient(patient_id, f"Your appointment has been delayed by {delay_minutes} minutes.")

    # Log the action with more details
    logger.info(f"Patient {patient_id} delayed by {delay_minutes} minutes. Base Wait Time: {delayed_patient['base_wait_time']:.1f}, Cumulative Delay: {delayed_patient['cumulative_delay']:.1f}")

    # Recalculate wait times for all patients
    recalculate_wait_times()

    print_queue()
    return queue

def handle_done_early(patient_id, actual_time_taken):
    """Handle a patient finishing early and update the queue"""
    global queue
    if patient_id not in queue:
        print(f"Patient {patient_id} not found in queue.")
        return queue

    completed_patient = queue[patient_id]
    completed_position = completed_patient['queue_position']
    predicted_wait_time = completed_patient.get('predicted_wait_time', 0)  # Fallback to 0 if not set
    
    # Calculate time saved
    time_saved = predicted_wait_time - actual_time_taken
    if time_saved < 0:
        print(f"Patient {patient_id} took longer than predicted ({actual_time_taken} vs {predicted_wait_time} minutes). Treating as a delay.")
        # Treat as a delay if the patient took longer than predicted
        handle_delay(patient_id, -time_saved)  # Negative time_saved means a delay
        return queue

    print(f"Patient {patient_id} finished early. Predicted: {predicted_wait_time:.1f} minutes, Actual: {actual_time_taken:.1f} minutes, Time Saved: {time_saved:.1f} minutes.")
    
    # Notify the completed patient
    notify_patient(patient_id, f"Your appointment is complete. You finished {time_saved:.1f} minutes earlier than expected.")

    # Log the action with more details
    logger.info(f"Patient {patient_id} finished early. Predicted: {predicted_wait_time:.1f} minutes, Actual: {actual_time_taken:.1f} minutes, Time Saved: {time_saved:.1f} minutes.")

    # Remove the completed patient from the queue
    del queue[patient_id]

    # Update queue positions and lengths for remaining patients
    for pid, patient in queue.items():
        if patient['queue_position'] > completed_position:
            patient['queue_position'] -= 1
        patient['queue_length'] -= 1
        # Apply the time saved by reducing the cumulative delay
        if patient['queue_position'] >= completed_position:
            patient['cumulative_delay'] -= time_saved
            patient['cumulative_delay'] = max(0, patient['cumulative_delay'])
            print(f"Patient {pid}: Reduced cumulative delay by {time_saved:.1f} minutes to {patient['cumulative_delay']:.1f} minutes due to Patient {patient_id} finishing early.")
            logger.info(f"Patient {pid} cumulative delay reduced by {time_saved:.1f} minutes. New Cumulative Delay: {patient['cumulative_delay']:.1f}")

    # Recalculate wait times for all remaining patients
    recalculate_wait_times()

    print_queue()
    return queue

def handle_cancellation(patient_id):
    """Cancel a patient's appointment and update the queue with resource reallocation"""
    global queue
    if patient_id not in queue:
        print(f"Patient {patient_id} not found in queue.")
        return queue

    canceled_position = queue[patient_id]['queue_position']
    print(f"Patient {patient_id}'s appointment canceled.")
    
    # Notify patient of cancellation
    notify_patient(patient_id, "Your appointment has been canceled.")

    # Log the action
    logger.info(f"Patient {patient_id}'s appointment canceled.")

    del queue[patient_id]

    # Update queue positions and lengths for remaining patients
    for pid, patient in queue.items():
        if patient['queue_position'] > canceled_position:
            patient['queue_position'] -= 1
        patient['queue_length'] -= 1

    # Recalculate wait times for all remaining patients
    recalculate_wait_times()

    print_queue()
    return queue

def add_feedback(patient_id, actual_wait_time):
    """Add feedback for a patient"""
    global queue, feedback_data
    if patient_id not in queue:
        print(f"Patient {patient_id} not found in queue.")
        return feedback_data

    patient_data = queue[patient_id].copy()
    # Anonymize sensitive data before storing feedback
    patient_data.pop('patient_age', None)  # Remove sensitive info
    patient_data['actual_wait_time'] = actual_wait_time
    feedback_data.append(patient_data)
    print(f"Feedback added for patient {patient_id}: Actual wait time: {actual_wait_time} minutes.")

    # Log the action
    logger.info(f"Feedback added for Patient {patient_id}: Actual wait time {actual_wait_time} minutes.")

    return feedback_data

def retrain_model():
    """Retrain the model with feedback data using incremental learning"""
    global model_pipeline, explainer, feedback_data
    if not feedback_data:
        print("No feedback data to retrain the model.")
        return

    feedback_df = pd.DataFrame(feedback_data)
    feedback_df = engineer_features(feedback_df)
    X_feedback = feedback_df[raw_features]
    y_feedback = feedback_df['actual_wait_time']

    # Incremental update: Add more trees to the existing model
    model_pipeline.named_steps['model'].set_params(warm_start=True)
    model_pipeline.named_steps['model'].n_estimators += 10  # Add more trees incrementally
    model_pipeline.fit(X_feedback, y_feedback)

    try:
        xgb_model = model_pipeline.named_steps['model']
        explainer = shap.TreeExplainer(xgb_model)
        print("Using TreeExplainer for SHAP explanations after retraining.")
    except Exception as e:
        print(f"TreeExplainer failed after retraining: {e}. Falling back to KernelExplainer.")
        background_data = synthetic_data[raw_features].head(100)
        explainer = shap.KernelExplainer(model_pipeline.predict, background_data)

    joblib.dump(model_pipeline, '../models/wait_time_model.pkl')
    print("Model retrained with feedback data.")

    # Log the action
    logger.info("Model retrained with feedback data.")

def print_queue():
    """Print the current queue"""
    global queue
    if not queue:
        print("\nQueue is empty.")
        return

    queue_df = pd.DataFrame(list(queue.values()))
    print("\nCurrent Queue:")
    print(queue_df[['patient_id', 'department', 'predicted_wait_time', 'notified_wait_time', 'queue_position', 'queue_length', 'is_follow_up']].to_string(index=False))

# Flask app for web interface
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your-secret-key'  # Required for session management
CORS(app)  # Enable CORS for all routes

# Debug template folder path
print("Template folder path:", app.template_folder)
print("Absolute template folder path:", os.path.abspath(app.template_folder))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':  # Replace with secure authentication
            session['logged_in'] = True
            return redirect(url_for('show_queue'))
        else:
            flash("Invalid credentials", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/')
def show_queue():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Prepare queue data and ensure predicted_wait_time and priority_score are set
    queue_data = [
        {'patient_id': pid, **patient}
        for pid, patient in queue.items()
    ]
    
    # Ensure all patients have a predicted_wait_time, notified_wait_time, and priority_score
    for patient in queue_data:
        if 'predicted_wait_time' not in patient or patient['predicted_wait_time'] is None:
            patient['predicted_wait_time'] = 0.0  # Fallback value
            logger.warning(f"Patient {patient['patient_id']} had missing predicted_wait_time. Set to 0.0.")
        if 'notified_wait_time' not in patient or patient['notified_wait_time'] is None:
            patient['notified_wait_time'] = patient['predicted_wait_time']
            logger.warning(f"Patient {patient['patient_id']} had missing notified_wait_time. Set to {patient['notified_wait_time']:.1f}.")
        if 'priority_score' not in patient or patient['priority_score'] is None:
            # Calculate priority_score if missing
            patient['priority_score'] = patient['patient_condition_severity'] * 0.8
            logger.warning(f"Patient {patient['patient_id']} had missing priority_score. Calculated as {patient['priority_score']}.")
        if 'is_follow_up' not in patient:
            patient['is_follow_up'] = 0  # Fallback to 0 (not a follow-up)
            logger.warning(f"Patient {patient['patient_id']} had missing is_follow_up. Set to 0.")

    # Sort by queue_position for display
    queue_data.sort(key=lambda x: x.get('queue_position', 0))
    
    # Log queue data for debugging
    logger.info(f"Rendering queue with {len(queue_data)} patients: {queue_data}")
    
    return render_template('queue.html', queue=queue_data)

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            patient_data = {
                'patient_id': int(request.form['patient_id']),
                'name': request.form.get('name', f"Patient {request.form['patient_id']}"),
                'department': request.form['department'],
                'patient_condition_severity': int(request.form['patient_condition_severity']),
                'day_of_week': int(request.form['day_of_week']),
                'hour': int(request.form['hour']),
                'queue_length': int(request.form['queue_length']),
                'doctors_available': int(request.form['doctors_available']),
                'queue_position': int(request.form['queue_position']),
                'patient_age': int(request.form['patient_age']),
                'delay_factor': int(request.form['delay_factor']),
                'is_canceled': int(request.form['is_canceled']),
                'doctor_experience': int(request.form['doctor_experience']),
                'historical_wait_time': int(request.form['historical_wait_time']),
                'is_follow_up': int(request.form['is_follow_up']),
            }

            # Input validation
            if patient_data['patient_id'] <= 0:
                flash("Patient ID must be positive.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['patient_condition_severity'] < 1 or patient_data['patient_condition_severity'] > 5:
                flash("Condition severity must be between 1 and 5.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['day_of_week'] < 0 or patient_data['day_of_week'] > 6:
                flash("Day of week must be between 0 and 6.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['hour'] < 0 or patient_data['hour'] > 23:
                flash("Hour must be between 0 and 23.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['queue_length'] < 1:
                flash("Queue length must be at least 1.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['doctors_available'] < 1:
                flash("Doctors available must be at least 1.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['queue_position'] < 1 or patient_data['queue_position'] > patient_data['queue_length']:
                flash("Queue position must be between 1 and queue length.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['patient_age'] < 1 or patient_data['patient_age'] > 120:
                flash("Patient age must be between 1 and 120.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['delay_factor'] < 0:
                flash("Delay factor cannot be negative.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['is_canceled'] not in [0, 1]:
                flash("Is canceled must be 0 or 1.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['doctor_experience'] < 1:
                flash("Doctor experience must be at least 1 year.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['historical_wait_time'] < 0:
                flash("Historical wait time cannot be negative.", "error")
                return redirect(url_for('add_patient'))
            if patient_data['is_follow_up'] not in [0, 1]:
                flash("Is follow-up must be 0 or 1.", "error")
                return redirect(url_for('add_patient'))

            update_queue(patient_data)
            flash(f"Patient {patient_data['patient_id']} added successfully.", "message")
            return redirect(url_for('show_queue'))
        except ValueError as e:
            flash("Invalid input: All fields must be numeric where required.", "error")
            return redirect(url_for('add_patient'))
    return render_template('add_patient.html')

@app.route('/delay_patient', methods=['POST'])
def delay_patient():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        patient_id = int(request.form['patient_id'])
        delay_minutes = int(request.form['delay_minutes'])
        if delay_minutes <= 0:
            flash("Delay minutes must be positive.", "error")
            return redirect(url_for('show_queue'))
        handle_delay(patient_id, delay_minutes)
        flash(f"Patient {patient_id} delayed by {delay_minutes} minutes.", "message")
    except ValueError:
        flash("Invalid input: Patient ID and delay minutes must be numeric.", "error")
    return redirect(url_for('show_queue'))

@app.route('/missed_appointment', methods=['POST'])
def missed_appointment():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        patient_id = int(request.form['patient_id'])
        new_position_after_patient_id = int(request.form['new_position_after_patient_id'])
        handle_missed_appointment(patient_id, new_position_after_patient_id)
        flash(f"Patient {patient_id} has been moved to after Patient {new_position_after_patient_id}.", "message")
    except ValueError:
        flash("Invalid input: Patient IDs must be numeric.", "error")
    return redirect(url_for('show_queue'))

@app.route('/done_early', methods=['POST'])
def done_early():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        patient_id = int(request.form['patient_id'])
        actual_time_taken = float(request.form['actual_time_taken'])
        if actual_time_taken < 0:
            flash("Actual time taken cannot be negative.", "error")
            return redirect(url_for('show_queue'))
        handle_done_early(patient_id, actual_time_taken)
        flash(f"Patient {patient_id} finished early. Time saved applied to queue.", "message")
    except ValueError:
        flash("Invalid input: Patient ID and actual time taken must be numeric.", "error")
    return redirect(url_for('show_queue'))

@app.route('/cancel_patient', methods=['POST'])
def cancel_patient():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        patient_id = int(request.form['patient_id'])
        handle_cancellation(patient_id)
        flash(f"Patient {patient_id}'s appointment canceled.", "message")
    except ValueError:
        flash("Invalid input: Patient ID must be numeric.", "error")
    return redirect(url_for('show_queue'))

@app.route('/add_feedback', methods=['POST'])
def add_feedback_route():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        patient_id = int(request.form['patient_id'])
        actual_wait_time = float(request.form['actual_wait_time'])
        if actual_wait_time < 0:
            flash("Actual wait time cannot be negative.", "error")
            return redirect(url_for('show_queue'))
        add_feedback(patient_id, actual_wait_time)
        retrain_model()
        flash(f"Feedback added for Patient {patient_id}.", "message")
    except ValueError:
        flash("Invalid input: Patient ID and actual wait time must be numeric.", "error")
    return redirect(url_for('show_queue'))

@app.route('/api/queue', methods=['GET'])
def get_queue():
    queue_data = [
        {'patient_id': pid, **patient}
        for pid, patient in queue.items()
    ]
    queue_data.sort(key=lambda x: x.get('queue_position', 0))
    return jsonify(queue_data)

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient_wait_time(patient_id):
    if patient_id not in queue:
        return jsonify({'error': f'Patient {patient_id} not found in queue.'}), 404
    patient_data = queue[patient_id]
    return jsonify({
        'patient_id': patient_data['patient_id'],
        'name': patient_data.get('name', f"Patient {patient_data['patient_id']}"),
        'department': patient_data['department'],
        'queue_position': patient_data['queue_position'],
        'queue_length': patient_data['queue_length'],
        'predicted_wait_time': patient_data.get('predicted_wait_time', 0.0),
        'notified_wait_time': patient_data.get('notified_wait_time', 0.0),
        'base_wait_time': patient_data.get('base_wait_time', 0.0),
        'cumulative_delay': patient_data.get('cumulative_delay', 0.0),
        'appointment_time': patient_data.get('appointment_time', 'N/A'),
        'is_follow_up': patient_data.get('is_follow_up', 0)
    })

# Demo with patients matching React app
if __name__ == "__main__":
    # Define patients to match React app's StaffPanel
    patients = [
        {
            'patient_id': 1,
            'name': 'John Smith',
            'department': 'Emergency',
            'patient_condition_severity': 2,
            'status': 'waiting',
            'is_canceled': 0,
            'is_follow_up': 0,
        },
        {
            'patient_id': 2,
            'name': 'Sarah Johnson',
            'department': 'Emergency',
            'patient_condition_severity': 3,
            'status': 'in-progress',
            'is_canceled': 0,
            'is_follow_up': 1,
        },
        {
            'patient_id': 3,
            'name': 'Michael Brown',
            'department': 'Cardiology',
            'patient_condition_severity': 1,
            'status': 'waiting',
            'is_canceled': 0,
            'is_follow_up': 0,
        },
        {
            'patient_id': 4,
            'name': 'Emily Davis',
            'department': 'Cardiology',
            'patient_condition_severity': 2,
            'status': 'waiting',
            'is_canceled': 0,
            'is_follow_up': 1,
        },
        {
            'patient_id': 5,
            'name': 'Robert Wilson',
            'department': 'Emergency',
            'patient_condition_severity': 2,
            'status': 'completed',
            'is_canceled': 0,
            'is_follow_up': 0,
        },
        {
            'patient_id': 6,
            'name': 'Jennifer Lee',
            'department': 'Emergency',
            'patient_condition_severity': 4,
            'status': 'waiting',
            'is_canceled': 0,
            'is_follow_up': 1,
        },
        {
            'patient_id': 7,
            'name': 'David Miller',
            'department': 'General Practice',
            'patient_condition_severity': 1,
            'status': 'waiting',
            'is_canceled': 0,
            'is_follow_up': 0,
        },
        {
            'patient_id': 8,
            'name': 'Lisa Anderson',
            'department': 'Dermatology',
            'patient_condition_severity': 2,
            'status': 'cancelled',
            'is_canceled': 1,
            'is_follow_up': 1,
        },
    ]

    # Add common attributes to all patients
    for i, patient in enumerate(patients, 1):
        patient['day_of_week'] = datetime.now().weekday()
        patient['hour'] = datetime.now().hour
        patient['queue_length'] = 8  # Total number of patients
        patient['queue_position'] = i
        patient['doctors_available'] = 2
        patient['patient_age'] = 35
        patient['delay_factor'] = 0
        patient['doctor_experience'] = 5
        patient['historical_wait_time'] = 5 if patient['department'] == 'Emergency' else 30
        patient['appointment_time'] = (datetime.now() + timedelta(minutes=i * 10)).isoformat()

    # Add patients to the queue
    print("\n=== Adding Patients to the Queue ===")
    for i, patient in enumerate(patients, 1):
        print(f"\nAdding Patient {i}:")
        update_queue(patient)

    # Simulate Patient 2 missing their turn and arriving after Patient 4
    print("\n=== Simulating Missed Appointment ===")
    print("\nPatient 2 misses their turn and arrives after Patient 4:")
    handle_missed_appointment(patient_id=2, new_position_after_patient_id=4)

    # Simulate Patient 1 finishing early to demonstrate reduced idle time
    print("\n=== Simulating Patient 1 Finishing Early ===")
    print("\nPatient 1 finishes early (actual time: 5 minutes):")
    handle_done_early(patient_id=1, actual_time_taken=5)

    # Simulate delays for some patients
    print("\n=== Simulating Delays ===")
    print("\nSimulating a delay for Patient 3 (10 minutes):")
    handle_delay(patient_id=3, delay_minutes=10)

    # Add feedback for some patients
    print("\n=== Adding Feedback ===")
    print("\nAdding feedback for Patient 3 (actual wait time: 30 minutes):")
    add_feedback(patient_id=3, actual_wait_time=30)

    # Retrain the model with feedback data
    print("\n=== Retraining Model with Feedback Data ===")
    retrain_model()

    # Check the audit log file
    print("\n=== Checking Audit Log ===")
    try:
        with open('queue_audit.log', 'r') as log_file:
            log_contents = log_file.readlines()
        print(f"Audit log contains {len(log_contents)} entries. Last 5 entries:")
        for line in log_contents[-5:]:
            print(line.strip())
    except FileNotFoundError:
        print("Error: Audit log file 'queue_audit.log' not found. Logging failed.")

    # Final queue state
    print("\n=== Final Queue State ===")
    print_queue()

    # Start the Flask app
    print("\nStarting web interface at http://127.0.0.1:5000")
    app.run(debug=True)