import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('../data', exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../plots', exist_ok=True)

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_synthetic_data(n_patients=5000, save_path='../data/synthetic_data.csv'):
    """Generate simplified synthetic hospital wait time data"""
    logger.info(f"Generating synthetic data for {n_patients} patients")

    # Define departments and hours
    departments = np.random.choice(
        ['Emergency', 'General', 'Cardiology', 'Pediatrics', 'Orthopedics'],
        size=n_patients, p=[0.4, 0.3, 0.15, 0.1, 0.05]
    )
    hours = np.random.randint(0, 24, size=n_patients)
    day_of_week = np.random.randint(0, 7, size=n_patients)

    # Generate appointment dates
    base_date = datetime(2025, 3, 22)
    appointment_dates = [
        base_date + timedelta(
            days=int(np.random.randint(0, 30)),  # Convert to Python int
            hours=int(hour),                     # Convert to Python int
            minutes=int(np.random.randint(0, 60))  # Convert to Python int
        )
        for hour in hours
    ]
    appointment_dates.sort()

    # Simplified queue length and doctors available
    queue_length = np.random.randint(1, 40, n_patients)
    doctors_available = np.random.randint(1, 6, n_patients)

    # Patient condition severity and age
    patient_condition_severity = np.random.randint(1, 5, n_patients)
    patient_age = np.random.randint(1, 90, n_patients)

    # Simplified additional features
    doctor_experience = np.random.randint(1, 30, n_patients)
    historical_wait_time = np.random.randint(10, 60, n_patients)
    delay_factor = np.random.randint(0, 10, n_patients)
    queue_position = np.array([np.random.randint(1, ql + 1) for ql in queue_length])
    is_follow_up = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])  # 40% follow-up patients

    data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'department': departments,
        'appointment_time': [d.isoformat() for d in appointment_dates],
        'hour': hours,
        'day_of_week': day_of_week,
        'queue_length': queue_length,
        'doctors_available': doctors_available,
        'patient_condition_severity': patient_condition_severity,
        'doctor_experience': doctor_experience,
        'patient_age': patient_age,
        'historical_wait_time': historical_wait_time,
        'queue_position': queue_position,
        'is_canceled': np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
        'delay_factor': delay_factor,
        'is_follow_up': is_follow_up,  # Added new feature
    })

    # Simplified wait time calculation, adjusted for follow-up
    data['wait_time'] = (
        (data['queue_position'] * 1.5) +
        (data['queue_length'] * 0.5) +
        (data['patient_condition_severity'] * 5) +
        (data['delay_factor'] * 2) -
        (data['doctors_available'] * 4) +
        (data['department'].apply(lambda x: 5 if x == 'Emergency' else 0)) -
        (data['is_follow_up'] * 3)  # Follow-up patients have slightly shorter wait times
    ) + np.random.normal(0, 5, n_patients)

    data['wait_time'] = np.maximum(data['wait_time'], 3)
    data['wait_time'] = np.minimum(data['wait_time'], 120)

    data.to_csv(save_path, index=False)
    logger.info(f"Synthetic dataset saved as '{save_path}'")
    return data

def engineer_features(data):
    """Simplified feature engineering"""
    data = data.copy()
    data['is_emergency'] = (data['department'] == 'Emergency').astype(int)
    data['is_specialized'] = data['department'].apply(
        lambda x: 1 if x in ['Cardiology', 'Orthopedics', 'Pediatrics'] else 0
    )
    data['priority_score'] = data['patient_condition_severity'] * 0.8
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['is_night_shift'] = data['hour'].apply(lambda x: 1 if x < 8 or x >= 20 else 0)
    data['is_peak_hour'] = data['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (14 <= x <= 17) else 0)
    data['time_of_day'] = data['hour'].apply(lambda x: 
        'morning' if 5 <= x < 12 else
        'afternoon' if 12 <= x < 17 else
        'evening' if 17 <= x < 22 else
        'night'
    )
    data['patients_per_doctor'] = data['queue_length'] / np.maximum(data['doctors_available'], 1)
    data['queue_efficiency'] = (data['queue_length'] - data['queue_position']) / np.maximum(data['queue_length'], 1)
    data['is_senior'] = (data['patient_age'] >= 65).astype(int)
    data['is_child'] = (data['patient_age'] <= 12).astype(int)
    data['emergency_severity'] = data['is_emergency'] * data['patient_condition_severity']
    data['critical_case'] = (data['patient_condition_severity'] >= 3).astype(int)
    return data

def prepare_data_for_modeling(data, target='wait_time'):
    """Prepare data for modeling"""
    logger.info("Preparing data for modeling")

    numerical_features = [
        'hour', 'day_of_week', 'queue_length', 'is_emergency', 'priority_score',
        'doctors_available', 'patient_condition_severity', 'delay_factor', 'is_canceled',
        'doctor_experience', 'patient_age', 'historical_wait_time', 'queue_position',
        'is_peak_hour', 'is_night_shift', 'is_specialized',
        'patients_per_doctor', 'queue_efficiency', 'is_senior', 'is_child',
        'emergency_severity', 'critical_case', 'is_weekend',
        'is_follow_up'  # Added new feature
    ]
    categorical_features = ['department', 'time_of_day']

    numerical_features = [f for f in numerical_features if f in data.columns]
    categorical_features = [f for f in categorical_features if f in data.columns]
    all_features = numerical_features + categorical_features

    X = data[all_features].copy()
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

def build_model_pipeline(numerical_features, categorical_features):
    """Build a simplified model pipeline"""
    logger.info("Building model pipeline")

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_SEED
        ))
    ])
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    logger.info("Evaluating model performance")

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {'RMSE': rmse, 'MAE': mae, 'R-squared': r2}
    logger.info(f"Model Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    return metrics, y_pred

def plot_feature_importance(model, numerical_features, categorical_features, save_path='../plots/feature_importance.png'):
    """Simplified feature importance plotting"""
    logger.info("Plotting feature importance")

    xgb_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

    importances = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")

    return feature_importance

def plot_actual_vs_predicted(y_test, y_pred, save_path='../plots/actual_vs_predicted.png'):
    """Plot actual vs predicted values"""
    logger.info("Plotting actual vs predicted values")

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Wait Time (minutes)')
    plt.ylabel('Predicted Wait Time (minutes)')
    plt.title('Actual vs Predicted Wait Times')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Actual vs predicted plot saved to {save_path}")

def save_model_results(model, metrics, feature_importance, save_dir='../models'):
    """Save model and results"""
    logger.info("Saving model and results")

    joblib.dump(model, os.path.join(save_dir, 'wait_time_model.pkl'))
    with open(os.path.join(save_dir, 'model_metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    feature_importance.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False)

def main():
    """Main function to run the modeling pipeline"""
    try:
        # For real hospital integration, replace this with actual data from an EHR system
        # Example: Fetch data using an API (e.g., FHIR) or database query
        # data = fetch_data_from_ehr()
        # Map EHR data to required features:
        # - patient_condition_severity: Map to triage score (e.g., 1-5)
        # - doctors_available: Fetch from staff schedule
        # - queue_length: Count patients in queue per department
        data = generate_synthetic_data(n_patients=5000)
        data = engineer_features(data)
        X_train, X_test, y_train, y_test, numerical_features, categorical_features = prepare_data_for_modeling(data)
        pipeline = build_model_pipeline(numerical_features, categorical_features)
        pipeline.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
        feature_importance = plot_feature_importance(pipeline, numerical_features, categorical_features)
        plot_actual_vs_predicted(y_test, y_pred)
        save_model_results(pipeline, metrics, feature_importance)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in modeling pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()