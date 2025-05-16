import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

def validate_input(data):
    required_fields = ['country', 'amount', 'transaction_type']
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    try:
        amount = float(data['amount'])
        if amount <= 0:
            return False, "Amount must be positive"
    except ValueError:
        return False, "Invalid amount"
    
    return True, "Validation passed"

def generate_visualization(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='prediction_result', data=data)
    plt.title('Transaction Results Distribution')
    plt.xlabel('Result')
    plt.ylabel('Count')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def create_and_save_model():
    # Define preprocessing steps
    numeric_features = ['Amount', 'Money Laundering Risk Score', 'Shell Companies Involved']
    numeric_transformer = StandardScaler()

    categorical_features = ['Country', 'Transaction Type', 'Person Involved', 'Industry', 
                          'Destination Country', 'Financial Institution', 'Tax Haven Country']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Create dummy data to train the model
    data = {
        'Country': ['USA', 'UK', 'Panama', 'USA', 'UK', 'Panama'],
        'Amount': [1000, 5000, 100000, 2000, 3000, 50000],
        'Transaction Type': ['Transfer', 'Withdrawal', 'Deposit', 'Transfer', 'Withdrawal', 'Deposit'],
        'Person Involved': ['John', 'Alice', 'Unknown', 'John', 'Alice', 'Unknown'],
        'Industry': ['Banking', 'Retail', 'Unknown', 'Banking', 'Retail', 'Unknown'],
        'Destination Country': ['UK', 'USA', 'Bermuda', 'UK', 'USA', 'Bermuda'],
        'Money Laundering Risk Score': [2, 3, 9, 3, 4, 8],
        'Shell Companies Involved': [0, 0, 1, 0, 0, 1],
        'Financial Institution': ['Bank A', 'Bank B', 'Unknown', 'Bank A', 'Bank B', 'Unknown'],
        'Tax Haven Country': ['None', 'None', 'Bermuda', 'None', 'None', 'Bermuda'],
        'Transaction_Year': [2023, 2023, 2023, 2023, 2023, 2023],
        'Transaction_Month': [1, 1, 1, 2, 2, 2],
        'Transaction_Day': [1, 2, 3, 4, 5, 6],
        'Transaction_DayOfWeek': [1, 2, 3, 4, 5, 6],
        'Transaction_Hour': [10, 12, 14, 10, 12, 14],
        'Reported by Authority': [False, False, True, False, False, True],
        'Is Illegal': [0, 0, 1, 0, 0, 1]
    }

    df = pd.DataFrame(data)
    X = df.drop('Is Illegal', axis=1)
    y = df['Is Illegal']

    # Train the model
    model.fit(X, y)

    # Save the model and preprocessor
    model_path = os.path.join('models', 'hybrid_model.pkl')
    joblib.dump(model, model_path)

    return model

def load_model():
    model_path = os.path.join('models', 'hybrid_model.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        # Create and save a new model if it doesn't exist
        return create_and_save_model()

def retrain_model(new_data_path):
    """Retrain the model with new data"""
    try:
        # Load existing model
        model = load_model()
        
        # Load new data
        new_data = pd.read_csv(new_data_path)
        
        # Validate new data
        required_columns = ['Country', 'Amount', 'Transaction Type', 'Is Illegal']
        missing_cols = [col for col in required_columns if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"New data missing required columns: {', '.join(missing_cols)}")
        
        # Add missing columns with default values if needed
        for col in ['Person Involved', 'Industry', 'Destination Country']:
            if col not in new_data.columns:
                new_data[col] = 'Unknown'
        
        if 'Money Laundering Risk Score' not in new_data.columns:
            new_data['Money Laundering Risk Score'] = 5
        
        if 'Shell Companies Involved' not in new_data.columns:
            new_data['Shell Companies Involved'] = 0
        
        # Add datetime features
        now = datetime.now()
        new_data['Transaction_Year'] = now.year
        new_data['Transaction_Month'] = now.month
        new_data['Transaction_Day'] = now.day
        new_data['Transaction_DayOfWeek'] = now.weekday()
        new_data['Transaction_Hour'] = now.hour
        new_data['Reported by Authority'] = False
        
        # Prepare data for retraining
        X_new = new_data.drop('Is Illegal', axis=1)
        y_new = new_data['Is Illegal']
        
        # Retrain the model
        model.fit(X_new, y_new)
        
        # Save the updated model
        model_path = os.path.join('models', 'hybrid_model.pkl')
        joblib.dump(model, model_path)
        
        return True, "Model retrained successfully"
    
    except Exception as e:
        return False, f"Error retraining model: {str(e)}"

def evaluate_model(test_data_path):
    """Evaluate model performance on test data"""
    try:
        # Load model
        model = load_model()
        
        # Load test data
        test_data = pd.read_csv(test_data_path)
        
        # Validate test data
        required_columns = ['Country', 'Amount', 'Transaction Type', 'Is Illegal']
        missing_cols = [col for col in required_columns if col not in test_data.columns]
        if missing_cols:
            raise ValueError(f"Test data missing required columns: {', '.join(missing_cols)}")
        
        # Prepare test data
        X_test = test_data.drop('Is Illegal', axis=1)
        y_test = test_data['Is Illegal']
        
        # Make predictions
        from sklearn.metrics import classification_report, accuracy_score
        predictions = model.predict(X_test)
        
        
        # Generate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importances': get_feature_importances(model)
        }
    
    except Exception as e:
        return {'error': str(e)}

def get_feature_importances(model):
    """Extract feature importances from the trained model"""
    try:
        # Get feature names from the preprocessor
        numeric_features = ['Amount', 'Money Laundering Risk Score', 'Shell Companies Involved']
        categorical_features = ['Country', 'Transaction Type', 'Person Involved', 'Industry', 
                              'Destination Country', 'Financial Institution', 'Tax Haven Country']
        
        # Get the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Get feature importances
        importances = classifier.feature_importances_
        
        # Create a DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame({
            'feature': numeric_features + categorical_features,
            'importance': importances[:len(numeric_features) + len(categorical_features)]
        })
        
        return feature_importance_df.sort_values('importance', ascending=False).to_dict('records')
    
    except Exception as e:
        return {'error': str(e)}

def get_model_metadata():
    """Get metadata about the current model"""
    model_path = os.path.join('models', 'hybrid_model.pkl')
    
    if not os.path.exists(model_path):
        return {'status': 'Model not found', 'exists': False}
    
    model = joblib.load(model_path)
    
    return {
        'status': 'Model loaded',
        'exists': True,
        'model_type': str(type(model.named_steps['classifier'])),
        'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
        'model_size': f"{os.path.getsize(model_path) / 1024:.2f} KB"
    }