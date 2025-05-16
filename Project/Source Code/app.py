import os
import sqlite3
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import random
from threading import Thread
from uuid import uuid4
from model import load_model, validate_input, generate_visualization

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_FOLDER = 'models'
DB_NAME = 'transactions.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'developerharry18@gmail.com'
app.config['MAIL_PASSWORD'] = 'itst zeic zutz cknw'
app.config['MAIL_DEFAULT_SENDER'] = 'Tax_Evasion@gmail.com'

mail = Mail(app)

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize database
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            country TEXT,
            amount REAL,
            transaction_type TEXT,
            tax_amount INTEGER,
            prediction_result TEXT,
            confidence INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_email TEXT,
            notes TEXT
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            organization TEXT,
            subscription_type TEXT,
            last_login DATETIME,
            is_admin BOOLEAN DEFAULT 0
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_auth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            FOREIGN KEY (email) REFERENCES users(email)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            action TEXT,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_result TEXT,
            confidence INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

init_db()

# Load the model
model = load_model()

# Feature 1: Asynchronous email sending
def send_async_email(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            app.logger.error(f"Error sending email: {e}")

def send_email(subject, recipients, body):
    msg = Message(subject, recipients=recipients)
    msg.body = body
    Thread(target=send_async_email, args=(app, msg)).start()

# Feature 4: Log audit trail
def log_audit(user_email, action, details):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO audit_log (user_email, action, details)
        VALUES (?, ?, ?)
        ''', (user_email, action, details))
        conn.commit()

# Feature 5: Generate report
def generate_report():
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql('SELECT * FROM transactions', conn)
    
    if df.empty:
        return None
    
    report = {
        'total_transactions': len(df),
        'legal_count': len(df[df['prediction_result'] == 'Legal']),
        'illegal_count': len(df[df['prediction_result'] == 'Illegal']),
        'highest_amount': df['amount'].max(),
        'most_common_country': df['country'].mode()[0],
        'visualization': generate_visualization(df)
    }
    return report

# Feature 6: Save transaction to database
def save_transaction(data, prediction_result, confidence,tax_amount, user_email=None, notes=None ):
    transaction_id = str(uuid4())
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO transactions (
            transaction_id, country, amount, transaction_type,tax_amount,
            prediction_result, confidence, user_email, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction_id, data['country'], float(data['amount']), 
            data['transaction_type'],tax_amount, prediction_result, confidence, 
            user_email, notes
        ))
        conn.commit()
    return transaction_id

# Feature 7: User management
def add_user(email, name, organization, subscription_type='basic', is_admin=False):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO users (email, name, organization, subscription_type, is_admin)
            VALUES (?, ?, ?, ?, ?)
            ''', (email, name, organization, subscription_type, is_admin))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

# Feature 8: Transaction search
def search_transactions(search_term=None, start_date=None, end_date=None, result_type=None):
    query = 'SELECT * FROM transactions WHERE 1=1'
    params = []
    
    if search_term:
        query += ' AND (country LIKE ? OR transaction_type LIKE ?)'
        params.extend([f'%{search_term}%', f'%{search_term}%'])
    
    if start_date:
        query += ' AND timestamp >= ?'
        params.append(start_date)
    
    if end_date:
        query += ' AND timestamp <= ?'
        params.append(end_date)
    
    if result_type:
        query += ' AND prediction_result = ?'
        params.append(result_type)
    
    query += ' ORDER BY timestamp DESC LIMIT 100'
    
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(query, conn, params=params)
    
    return df

# Feature 9: Model performance monitoring
def log_prediction_metrics(prediction_result, confidence):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO model_metrics (prediction_result, confidence, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (prediction_result, confidence))
        conn.commit()

# Feature 10: Data export
def export_data(format='csv'):
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql('SELECT * FROM transactions', conn)
    
    if df.empty:
        return None
    
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'json':
        return df.to_json(orient='records')
    else:
        return None

# Tax calculation functions
def calculate_tax(income, age):
    tax = 0

    # Tax calculation for General Taxpayer
    if age < 60:
        if income <= 300000:
            tax = 0
        elif income <= 600000:
            tax = (income - 300000) * 0.05
        elif income <= 900000:
            tax = (income - 600000) * 0.10 + 15000  # 5% on ₹3L to ₹6L
        elif income <= 1200000:
            tax = (income - 900000) * 0.15 + 45000  # 10% on ₹6L to ₹9L
        elif income <= 1500000:
            tax = (income - 1200000) * 0.20 + 90000  # 15% on ₹9L to ₹12L
        else:
            tax = (income - 1500000) * 0.30 + 150000  # 20% on ₹12L to ₹15L
    
    # Tax calculation for Senior Citizens (60-80)
    elif 60 <= age < 80:
        if income <= 300000:
            tax = 0
        elif income <= 500000:
            tax = (income - 300000) * 0.05
        elif income <= 1000000:
            tax = (income - 500000) * 0.20 + 10000  # 5% on ₹3L to ₹5L
        else:
            tax = (income - 1000000) * 0.30 + 90000  # 20% on ₹5L to ₹10L
    
    # Tax calculation for Super Senior Citizens (80+)
    elif age >= 80:
        if income <= 500000:
            tax = 0
        elif income <= 1000000:
            tax = (income - 500000) * 0.20
        else:
            tax = (income - 1000000) * 0.30 + 100000  # 20% on ₹5L to ₹10L

    return tax

def get_indian_tax_brackets(age_group, year):
    """Indian tax brackets for FY 2023-24 (AY 2024-25)"""
    # Common brackets for all groups up to 50 years
    brackets = [
        (0, 300000, 0),          # 0% tax
        (300000, 600000, 0.05),   # 5% tax
        (600000, 900000, 0.10),   # 10% tax
        (900000, 1200000, 0.15),  # 15% tax
        (1200000, 1500000, 0.20), # 20% tax
        (1500000, float('inf'), 0.30) # 30% tax
    ]
    
    # Additional slabs for senior citizens (60-80 years)
    if age_group == 'senior_citizen':
        brackets = [
            (0, 300000, 0),          # 0% tax
            (300000, 500000, 0.05),   # 5% tax
            (500000, 1000000, 0.20),  # 20% tax
            (1000000, float('inf'), 0.30) # 30% tax
        ]
    
    # Additional slabs for super senior citizens (80+ years)
    elif age_group == 'super_senior':
        brackets = [
            (0, 500000, 0),           # 0% tax
            (500000, 1000000, 0.20),  # 20% tax
            (1000000, float('inf'), 0.30) # 30% tax
        ]
    
    return brackets

def calculate_indian_tax(income, brackets):
    """Calculate tax based on Indian tax brackets"""
    tax = 0
    for i, bracket in enumerate(brackets):
        lower, upper, rate = bracket
        if income > lower:
            if i == len(brackets) - 1:  # Last bracket
                taxable_in_bracket = income - lower
            else:
                taxable_in_bracket = min(income, upper) - lower
            tax += taxable_in_bracket * rate
    return tax

def calculate_rebate(tax_amount, taxable_income, age_group):
    """Calculate rebate under Section 87A"""
    rebate = 0
    if age_group in ['general', 'women']:
        if taxable_income <= 700000:  # Up to 7 lakhs
            rebate = min(tax_amount, 25000)
    elif age_group == 'senior_citizen':
        if taxable_income <= 750000:  # Up to 7.5 lakhs
            rebate = min(tax_amount, 25000)
    return rebate

def get_tax_breakdown(taxable_income, brackets):
    """Generate detailed tax breakdown"""
    breakdown = []
    for i, bracket in enumerate(brackets):
        lower, upper, rate = bracket
        if taxable_income > lower:
            if i == len(brackets) - 1:  # Last bracket
                amount = taxable_income - lower
            else:
                amount = min(taxable_income, upper) - lower
            tax = amount * rate
            breakdown.append({
                'range': "₹{:,.0f} - ₹{:,.0f}".format(lower, upper),
                'rate': "{:.0%}".format(rate),
                'amount': "₹{:,.0f}".format(amount),
                'tax': "₹{:,.0f}".format(tax)
            })
    return breakdown

# File upload helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from flask import current_app

def process_uploaded_file(filepath, user_email):
    try:
        # Load and preprocess data
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['Country', 'Amount', 'Transaction Type']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Add missing columns with default values
        for col in ['Person Involved', 'Industry', 'Destination Country']:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        if 'Money Laundering Risk Score' not in df.columns:
            df['Money Laundering Risk Score'] = 5
        
        if 'Shell Companies Involved' not in df.columns:
            df['Shell Companies Involved'] = 0
        
        # Add datetime features
        now = datetime.now()
        df['Transaction_Year'] = now.year
        df['Transaction_Month'] = now.month
        df['Transaction_Day'] = now.day
        df['Transaction_DayOfWeek'] = now.weekday()
        df['Transaction_Hour'] = now.hour
        df['Reported by Authority'] = False
        
        # Make predictions
        predictions = model.predict(df)
        probas = model.predict_proba(df)[:, 1]
        
        # Save results to database
        with sqlite3.connect(DB_NAME) as conn:
            for i, row in df.iterrows():
                confidence = probas[i] if predictions[i] == 1 else (1 - probas[i])
                
                if confidence >= 0.60:
                    result = "Legal"
                else:
                    result = "Illegal"
                
                # Fix the deprecated warning by using .iloc properly
                income = df.iloc[i, 2]  # Changed from [2] to ,2
                age = 45
                
                tax_amount = calculate_tax(income, age)
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO transactions (
                    transaction_id, country, amount, transaction_type, tax_amount,
                    prediction_result, confidence, user_email
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid4()), row['Country'], float(row['Amount']), 
                    row['Transaction Type'], tax_amount, result, confidence, user_email
                ))
            conn.commit()
        
        # Send completion email - using current_app within app context
        if user_email:
            with current_app.app_context():
                send_email(
                    "Bulk Transaction Processing Complete",
                    [user_email],
                    f"Your file {os.path.basename(filepath)} has been processed successfully."
                )
        
        # Log audit
        with current_app.app_context():
            log_audit(
                user_email or 'anonymous',
                'bulk_upload',
                f'Processed file {os.path.basename(filepath)} with {len(df)} transactions'
            )
        
    except Exception as e:
        # Log error and send error email within app context
        current_app.logger.error(f"Error processing uploaded file: {e}")
        if user_email:
            with current_app.app_context():
                send_email(
                    "Bulk Transaction Processing Failed",
                    [user_email],
                    f"An error occurred while processing your file {os.path.basename(filepath)}: {str(e)}"
                )
    finally:
        try:
            os.remove(filepath)
        except Exception as e:
            current_app.logger.error(f"Error removing temporary file: {e}")
# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # Get form data
        form_data = {
            'country': request.form.get('country'),
            'amount': request.form.get('amount'),
            'transaction_type': request.form.get('transaction_type'),
            'person_involved': request.form.get('person_involved'),
            'industry': request.form.get('industry'),
            'destination_country': request.form.get('destination_country'),
            'risk_score': request.form.get('risk_score'),
            'shell_companies': request.form.get('shell_companies'),
            'financial_institution': request.form.get('financial_institution'),
            'tax_haven': request.form.get('tax_haven'),
            'notes': request.form.get('notes')
        }
        
        # Validate input
        is_valid, message = validate_input(form_data)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('analyze'))
        
        # Create input DataFrame
        input_data = {
            'Country': [form_data['country']],
            'Amount': [float(form_data['amount'])],
            'Transaction Type': [form_data['transaction_type']],
            'Person Involved': [form_data.get('person_involved', 'Unknown')],
            'Industry': [form_data.get('industry', 'Unknown')],
            'Destination Country': [form_data.get('destination_country', 'Unknown')],
            'Money Laundering Risk Score': [int(form_data.get('risk_score', 5))],
            'Shell Companies Involved': [int(form_data.get('shell_companies', 0))],
            'Financial Institution': [form_data.get('financial_institution', 'Unknown')],
            'Tax Haven Country': [form_data.get('tax_haven', 'None')],
            'Transaction_Year': [datetime.now().year],
            'Transaction_Month': [datetime.now().month],
            'Transaction_Day': [datetime.now().day],
            'Transaction_DayOfWeek': [datetime.now().weekday()],
            'Transaction_Hour': [datetime.now().hour],
            'Reported by Authority': [False]
        }
        
        input_df = pd.DataFrame(input_data)
        
        try:
            # Make prediction
            pred = model.predict(input_df)
            proba = model.predict_proba(input_df)[0][1]
            
            confidence = 0
            if pred[0] == 1:
                
                confidence = proba
            else:
                
                confidence = 1 - proba
            
            
            if confidence >= 0.60:
                result = "Legal"
            else:
                result = "Illegal"
                
            income = float(form_data.get('amount', 0))
            age = int(request.form.get('age', 30))  # Default age if not provided
            tax_amount = calculate_tax(income, age)
            
            
            # Save transaction
            transaction_id = save_transaction(
                form_data, result, confidence,tax_amount,
                user_email=session.get('user_email'),
                notes=form_data.get('notes'),
                
            )
            
            # Log audit
            log_audit(
                session.get('user_email', 'anonymous'),
                'transaction_analysis',
                f'Analyzed transaction {transaction_id} with result {result}'
            )
            
            # Calculate tax (example - you would need to get income and age from form)
            
            
            # Prepare response
            response_data = {
                'transaction_id': transaction_id,
                'result': result,
                'tax_amount': f"₹{tax_amount:,.2f}",
                'confidence': f"{confidence:.2%}",
                'visualization': generate_visualization(pd.DataFrame({'prediction_result': [result]}))
            }
            
            # Send email notification if user is logged in
            if 'user_email' in session:
                email_body = f"""
                Your transaction analysis is complete:
                
                Transaction ID: {transaction_id}
                Result: {result}
                Confidence: {confidence:.2%}
                Estimated Tax: ₹{tax_amount:,.2f}
                
                Thank you for using our service.
                """
                send_email(
                    "Transaction Analysis Result",
                    [session['user_email']],
                    email_body
                )
            
            return render_template('result.html', data=response_data)
        
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            flash("An error occurred during analysis. Please try again.", 'error')
            return redirect(url_for('analyze'))
    
    return render_template('analyze.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            Thread(target=process_uploaded_file, args=(filepath, session.get('user_email'))).start()
            
            flash('File uploaded successfully. Processing will continue in the background.', 'success')
            return redirect(url_for('dashboard'))
    
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    # Get recent transactions
    with sqlite3.connect(DB_NAME) as conn:
        recent_transactions = pd.read_sql('''
        SELECT * FROM transactions 
        WHERE user_email = ?
        ORDER BY timestamp DESC 
        LIMIT 10
        ''', conn, params=(session['user_email'],))
    
    # Generate report
    report = generate_report()
    
    return render_template('dashboard.html', 
                         recent_transactions=recent_transactions.to_dict('records'),
                         report=report)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    result_type = request.args.get('result_type', '')
    
    results = search_transactions(query, start_date, end_date, result_type)
    
    return render_template('search.html', 
                         results=results.to_dict('records'),
                         query=query,
                         start_date=start_date,
                         end_date=end_date,
                         result_type=result_type)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        organization = request.form.get('organization', '')

        # Validate inputs
        if not all([email, name, password, confirm_password]):
            flash('All fields are required', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        # Check if email already exists
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT email FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                flash('Email already registered', 'error')
                return redirect(url_for('register'))

            # Create new user
            cursor.execute('''
                INSERT INTO users (email, name, organization, last_login)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (email, name, organization))
            conn.commit()

            # In a real app, you would hash the password before storing
            cursor.execute('''
                INSERT INTO user_auth (email, password_hash)
                VALUES (?, ?)
            ''', (email, password))  # In production, use generate_password_hash(password)
            conn.commit()

        flash('Registration successful! Please login.', 'success')
        log_audit(email, 'registration', 'New user registered')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # In a real app, you would verify the password hash
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT u.email, u.name, ua.password_hash 
                FROM users u
                JOIN user_auth ua ON u.email = ua.email
                WHERE u.email = ?
            ''', (email,))
            user = cursor.fetchone()
            
            if user and user[2] == password:  # In production, use check_password_hash()
                session['user_email'] = user[0]
                session['user_name'] = user[1]
                
                # Update last login
                cursor.execute('''
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE email = ?
                ''', (email,))
                conn.commit()
                
                # Log audit
                log_audit(email, 'login', 'User logged in')
                
                flash('Logged in successfully', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password', 'error')
                return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user_email' in session:
        # Log audit
        log_audit(session['user_email'], 'logout', 'User logged out')
        
        session.pop('user_email', None)
        session.pop('user_name', None)
    
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/tax-calculator', methods=['GET', 'POST'])
def tax_calculator():
    if request.method == 'POST':
        try:
            # Get form data with proper defaults
            income_lakhs = float(request.form.get('income', 0))
            age_group = request.form.get('age_group', 'general')  # Default to 'general'
            deductions = float(request.form.get('deductions', 0)) * 100000
            tax_year = int(request.form.get('tax_year', datetime.now().year))
            
            # Validate inputs
            if income_lakhs < 0:
                flash('Income cannot be negative', 'error')
                return redirect(url_for('tax_calculator'))
            
            income = income_lakhs * 100000  # Convert lakhs to rupees
            
            # Calculate taxable income after standard deduction
            standard_deduction = 50000
            taxable_income = max(0, income - deductions - standard_deduction)
            
            # Get tax brackets with proper age group handling
            brackets = get_indian_tax_brackets(age_group, tax_year)
            
            # Calculate tax amount
            tax_amount = calculate_indian_tax(taxable_income, brackets)
            
            # Calculate rebate under Section 87A
            rebate = calculate_rebate(tax_amount, taxable_income, age_group)
            tax_amount -= rebate
            
            # Add health and education cess (4%)
            
        
            cess = tax_amount * 0.04
            total_tax = tax_amount + cess
            
            # Calculate effective tax rate safely
            effective_rate = (total_tax / income) * 100 if income > 0 else 0
            
            # Prepare result with proper null checks
            result = {
                'gross_income': f"₹{income:,.2f} ({income/100000:.2f} Lakhs)",
                'deductions': f"₹{deductions:,.2f} ({deductions/100000:.2f} Lakhs)",
                'standard_deduction': "₹50,000",
                'taxable_income': f"₹{taxable_income:,.2f} ({taxable_income/100000:.2f} Lakhs)",
                'tax_amount': f"₹{total_tax:,.2f}",
                'effective_rate': f"{effective_rate:.2f}%",
                'age_group': age_group.replace('_', ' ').title() if age_group else 'General',
                'tax_year': tax_year,
                'tax_breakdown': get_tax_breakdown(taxable_income, brackets),
                'rebate': f"₹{rebate:,.2f}" if rebate > 0 else "₹0",
                'cess': f"₹{cess:,.2f}"
            }
            
            if 'user_email' in session:
                log_audit(
                    session['user_email'],
                    'tax_calculation',
                    f"Calculated tax: {result}"
                )
            
            return render_template('tax_result.html', result=result)
        
        except ValueError as e:
            app.logger.error(f"Tax calculation error: {str(e)}")
            flash('Please enter valid numbers', 'error')
            return redirect(url_for('tax_calculator'))
        except Exception as e:
            app.logger.error(f"Unexpected error in tax calculation: {str(e)}")
            flash('An error occurred during tax calculation', 'error')
            return redirect(url_for('tax_calculator'))
    
    current_year = datetime.now().year
    return render_template('tax_calculator.html', 
                         years=range(current_year, current_year - 5, -1))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate input
    is_valid, message = validate_input(data)
    if not is_valid:
        return jsonify({'error': message}), 400
    
    # Create input DataFrame
    input_data = {
        'Country': [data['country']],
        'Amount': [float(data['amount'])],
        'Transaction Type': [data['transaction_type']],
        'Person Involved': [data.get('person_involved', 'Unknown')],
        'Industry': [data.get('industry', 'Unknown')],
        'Destination Country': [data.get('destination_country', 'Unknown')],
        'Money Laundering Risk Score': [int(data.get('risk_score', 5))],
        'Shell Companies Involved': [int(data.get('shell_companies', 0))],
        'Financial Institution': [data.get('financial_institution', 'Unknown')],
        'Tax Haven Country': [data.get('tax_haven', 'None')],
        'Transaction_Year': [datetime.now().year],
        'Transaction_Month': [datetime.now().month],
        'Transaction_Day': [datetime.now().day],
        'Transaction_DayOfWeek': [datetime.now().weekday()],
        'Transaction_Hour': [datetime.now().hour],
        'Reported by Authority': [False]
    }
    
    input_df = pd.DataFrame(input_data)
    
    try:
        # Make prediction
        pred = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]
        
        #------------------------#
        confidence = 0
        if pred[0] == 1:
            
            confidence = proba
        else:
            
            confidence = 1 - proba
        
        
        if confidence >= 0.60:
            result = "Legal"
        else:
            result = "Illegal"
            
        income = 3000
        age = 30
        tax_amount = calculate_tax(income, age)
        
        # Save transaction
        transaction_id = save_transaction(
            data, result, confidence,tax_amount,
            user_email=session.get('user_email'),
            notes=data.get('notes'),tax_amount=0)
        
        # Log audit
        log_audit(
            session.get('user_email', 'api_user'),
            'api_prediction',
            f'API prediction for transaction {transaction_id}'
        )
        
        return jsonify({
            'transaction_id': transaction_id,
            'result': result,
            'confidence': confidence,
            'status': 'success'
        })
    
    except Exception as e:
        app.logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500






# Dummy data to simulate states and taxpayer statistics
COUNTRIES = ['India']
STATES = {
    'India': [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 
        'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 
        'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh', 
        'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry'
    ]
}

@app.route('/tax_find')
def tax_find():
    return render_template('tax_find.html')

# Endpoint to get states based on selected country
@app.route('/states')
def get_states():
    country = request.args.get('country')
    if country in STATES:
        return jsonify(STATES[country])
    return jsonify([])  # Return empty list if no states found

# Endpoint to generate random taxpayer statistics
@app.route('/tax-counts', methods=['POST'])
def get_tax_counts():
    data = request.get_json()
    country = data.get('country')
    state = data.get('state')

    if not country or not state:
        return jsonify({'error': 'Invalid country or state'})

    total_population = random.randint(400000, 9000000)  
    taxpayer_count = random.randint(int(total_population * 0.4), int(total_population * 0.7))  # Taxpayers (40% to 70% of the population)
    non_payer_count = total_population - taxpayer_count  

    response = {
        'filters': {
            'country': country,
            'state': state
        },
        'response': f'The total number of taxpayers in {state} is <strong>{taxpayer_count}</strong> and the number of non-payers is <strong>{non_payer_count}</strong> out of a total population of <strong>{total_population}</strong>.'
    }

    return jsonify(response)

    
if __name__ == '__main__':
    app.run(debug=True)