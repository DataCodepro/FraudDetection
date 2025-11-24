from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import io
import csv
from datetime import datetime
import os
import random
import time
import threading
from collections import deque
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'fraud-detection-system-secret-key-2025'

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_by TEXT DEFAULT 'system',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if not exists
    c.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role, created_by) 
        VALUES (?, ?, ?, ?, ?)
    ''', ('admin', 'admin@fraudsystem.com', generate_password_hash('admin123'), 'admin', 'system'))
    
    # Create default analyst user
    c.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role, created_by) 
        VALUES (?, ?, ?, ?, ?)
    ''', ('analyst', 'analyst@fraudsystem.com', generate_password_hash('user123'), 'user', 'system'))
    
    conn.commit()
    conn.close()

init_db()

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        if session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

class RealTimeFraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = []
        self.transaction_counter = 1000
        self.pending_transactions = deque(maxlen=100)
        self.checked_transactions = deque(maxlen=200)
        
    def generate_training_data(self):
        """Generate training data with realistic fraud patterns"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'transaction_id': [f'TX{i:06d}' for i in range(n_samples)],
            'amount': np.random.lognormal(5, 1.2, n_samples),
            'transaction_hour': np.random.randint(0, 24, n_samples),
            'transaction_day': np.random.randint(0, 7, n_samples),
            'category': np.random.choice(['Retail', 'Travel', 'Entertainment', 'Food & Dining', 'Electronics', 'Healthcare'], n_samples),
            'location': np.random.choice([
                'New York, US', 'Chicago, US', 'Los Angeles, US', 'Miami, US', 'San Francisco, US',
                'London, UK', 'Manchester, UK', 'Birmingham, UK',
                'Tokyo, JP', 'Osaka, JP', 'Kyoto, JP',
                'Paris, FR', 'Lyon, FR', 'Marseille, FR',
                'Berlin, DE', 'Munich, DE', 'Hamburg, DE',
                'Toronto, CA', 'Vancouver, CA', 'Montreal, CA',
                'Sydney, AU', 'Melbourne, AU', 'Brisbane, AU',
                'Singapore, SG', 'Hong Kong, HK', 'Dubai, AE',
                'Mumbai, IN', 'Delhi, IN', 'Bangalore, IN',
                'São Paulo, BR', 'Rio de Janeiro, BR',
                'Mexico City, MX', 'Buenos Aires, AR',
                'Online', 'International'
            ], n_samples),
            'device_type': np.random.choice(['Mobile App', 'Desktop Web', 'Tablet'], n_samples),
            'previous_transactions': np.random.poisson(20, n_samples),
            'avg_transaction_value': np.random.lognormal(4.5, 1, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'account_age_days': np.random.exponential(500, n_samples),
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(n_samples)]
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic fraud patterns
        fraud_conditions = (
            (df['amount'] > df['avg_transaction_value'] * 10) & 
            (df['transaction_hour'].between(0, 5)) &
            (df['location'].isin(['Online', 'International', 'Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN']))
        ) | (
            (df['amount'] > 2500) & 
            (df['category'] == 'Travel') &
            (df['previous_transactions'] < 15)
        ) | (
            (df['device_type'] == 'Mobile App') &
            (df['location'].isin(['Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN'])) &
            (df['transaction_hour'] < 6) &
            (df['amount'] > 800)
        ) | (
            (df['previous_transactions'] > 50) &
            (df['amount'] < 15) &
            (df['transaction_hour'].between(0, 4))
        ) | (
            (df['amount'] > 1500) &
            (df['account_age_days'] < 30) &
            (df['previous_transactions'] < 5)
        ) | (
            (df['category'] == 'Electronics') &
            (df['amount'] > 1200) &
            (df['transaction_hour'].between(22, 4))
        )
        
        df['is_fraud'] = fraud_conditions.astype(int)
        
        # Increase fraud rate
        fraud_rate_target = 0.14
        current_fraud_rate = df['is_fraud'].mean()
        
        if current_fraud_rate < fraud_rate_target:
            non_fraud_indices = df[df['is_fraud'] == 0].index.tolist()
            additional_frauds_needed = int((fraud_rate_target - current_fraud_rate) * len(df))
            
            if additional_frauds_needed > 0 and len(non_fraud_indices) > additional_frauds_needed:
                suspicious_candidates = df.loc[non_fraud_indices]
                high_amount_candidates = suspicious_candidates[suspicious_candidates['amount'] > 1000].index.tolist()
                international_candidates = suspicious_candidates[suspicious_candidates['location'].isin([
                    'Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN', 'Online', 'International'
                ])].index.tolist()
                new_account_candidates = suspicious_candidates[suspicious_candidates['account_age_days'] < 60].index.tolist()
                late_night_candidates = suspicious_candidates[suspicious_candidates['transaction_hour'].between(0, 5)].index.tolist()
                
                all_candidates = list(set(
                    high_amount_candidates + international_candidates + 
                    new_account_candidates + late_night_candidates
                ))
                
                if len(all_candidates) > additional_frauds_needed:
                    selected_frauds = random.sample(all_candidates, additional_frauds_needed)
                else:
                    selected_frauds = all_candidates + random.sample(
                        non_fraud_indices, 
                        additional_frauds_needed - len(all_candidates)
                    )
                
                df.loc[selected_frauds, 'is_fraud'] = 1
    
        fraud_rate = df['is_fraud'].mean()
        print(f"Generated training data with {df['is_fraud'].sum()} fraud cases ({fraud_rate:.2%})")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        df_processed = df.copy()
        
        categorical_columns = ['category', 'location', 'device_type']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        self.feature_columns = [
            'amount', 'transaction_hour', 'transaction_day', 'category',
            'location', 'device_type', 'previous_transactions', 
            'avg_transaction_value', 'customer_age', 'account_age_days'
        ]
        
        X = df_processed[self.feature_columns]
        y = df_processed['is_fraud']
        
        return X, y
    
    def train_model(self, df=None):
        """Train the Random Forest model"""
        try:
            if df is None:
                df = self.generate_training_data()
            
            X, y = self.prepare_features(df)
            
            print(f"Training on {len(X)} samples with {y.sum()} fraud cases ({y.mean():.2%})")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            print("Model training completed!")
            print(f"Train Accuracy: {train_accuracy:.2%}")
            print(f"Test Accuracy: {test_accuracy:.2%}")
            
            return {
                'feature_columns': self.feature_columns,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'feature_importance': feature_importance,
                'fraud_count': y.sum(),
                'total_samples': len(y),
                'fraud_rate': f"{y.mean():.2%}"
            }
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise e
    
    def generate_random_transaction(self):
        """Generate a random transaction for testing"""
        categories = ['Retail', 'Travel', 'Entertainment', 'Food & Dining', 'Electronics', 'Healthcare']
        locations = [
            'New York, US', 'Chicago, US', 'Los Angeles, US', 'Miami, US', 'San Francisco, US',
            'London, UK', 'Manchester, UK', 'Birmingham, UK',
            'Tokyo, JP', 'Osaka, JP', 'Kyoto, JP',
            'Paris, FR', 'Lyon, FR', 'Marseille, FR',
            'Berlin, DE', 'Munich, DE', 'Hamburg, DE',
            'Toronto, CA', 'Vancouver, CA', 'Montreal, CA',
            'Sydney, AU', 'Melbourne, AU', 'Brisbane, AU',
            'Singapore, SG', 'Hong Kong, HK', 'Dubai, AE',
            'Mumbai, IN', 'Delhi, IN', 'Bangalore, IN',
            'São Paulo, BR', 'Rio de Janeiro, BR',
            'Mexico City, MX', 'Buenos Aires, AR',
            'Online', 'International'
        ]
        devices = ['Mobile App', 'Desktop Web', 'Tablet']
        
        transaction = {
            'transaction_id': f'TRX-2025-{self.transaction_counter:04d}',
            'amount': round(np.random.lognormal(5, 1.5), 2),
            'transaction_hour': random.randint(0, 23),
            'transaction_day': random.randint(0, 6),
            'category': random.choice(categories),
            'location': random.choice(locations),
            'device_type': random.choice(devices),
            'previous_transactions': random.randint(1, 100),
            'avg_transaction_value': round(np.random.lognormal(4.5, 1), 2),
            'customer_age': random.randint(18, 80),
            'account_age_days': random.randint(1, 2000),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.transaction_counter += 1
        
        # Create fraud patterns
        is_high_fraud_risk = (
            (transaction['amount'] > 1500) or
            (transaction['amount'] > 800 and transaction['transaction_hour'] in [0, 1, 2, 3, 4, 5]) or
            (transaction['location'] in ['Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN', 'Online', 'International']) or
            (transaction['device_type'] == 'Mobile App' and transaction['amount'] > 1000 and transaction['previous_transactions'] < 10)
        )
        
        if is_high_fraud_risk and random.random() < 0.6:
            transaction['amount'] = transaction['amount'] * random.uniform(1.5, 3.0)
            if transaction['previous_transactions'] > 5:
                transaction['previous_transactions'] = random.randint(1, 5)
            if random.random() < 0.7:
                transaction['transaction_hour'] = random.choice([0, 1, 2, 3, 4, 5])
        
        if transaction['amount'] > 10000:
            transaction['amount'] = 10000
        
        return transaction
    
    def add_pending_transaction(self, transaction):
        """Add transaction to pending queue"""
        self.pending_transactions.append(transaction)
        return len(self.pending_transactions)
    
    def get_pending_transactions(self):
        """Get all pending transactions"""
        return list(self.pending_transactions)
    
    def clear_pending_transactions(self):
        """Clear all pending transactions"""
        self.pending_transactions.clear()
    
    def check_pending_transactions(self):
        """Check all pending transactions for fraud"""
        results = []
        transactions_to_process = list(self.pending_transactions)
        self.pending_transactions.clear()
        
        for transaction in transactions_to_process:
            result = self.predict_fraud(transaction)
            if 'error' not in result:
                full_result = {**transaction, **result}
                self.checked_transactions.append(full_result)
                results.append(full_result)
        
        return results
    
    def predict_fraud(self, transaction_data):
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            return {'error': 'Model not trained. Please train the model first.'}
        
        try:
            df = pd.DataFrame([transaction_data])
            
            for col in self.feature_columns:
                if col in self.label_encoders and col in df.columns:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        df[col] = 0
            
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            df = df[self.feature_columns]
            features = self.scaler.transform(df)
            
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            fraud_prob = probability[1]
            reason = self._generate_fraud_reason(transaction_data, fraud_prob)
            
            return {
                'is_fraud': bool(prediction),
                'fraud_probability': float(probability[1]),
                'confidence': float(max(probability)) * 100,
                'anomaly_score': float(probability[1]),
                'reason': reason,
                'status': 'FRAUD' if prediction else 'SAFE',
                'checked_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def _generate_fraud_reason(self, transaction, fraud_prob):
        """Generate human-readable fraud reason"""
        reasons = []
        
        if fraud_prob > 0.7:
            if transaction['amount'] > 1500:
                reasons.append("Unusually high transaction amount")
            if transaction['transaction_hour'] in [0, 1, 2, 3, 4, 5]:
                reasons.append("Transaction during unusual hours (midnight-6AM)")
            if transaction['location'] in ['Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN', 'Online', 'International']:
                reasons.append("International transaction from high-risk location")
            if transaction['device_type'] == 'Mobile App' and transaction['amount'] > 500:
                reasons.append("Large mobile transaction")
            if transaction['previous_transactions'] < 5 and transaction['amount'] > 300:
                reasons.append("New customer with large transaction")
    
        elif fraud_prob > 0.4:
            if transaction['amount'] > transaction['avg_transaction_value'] * 3:
                reasons.append("Amount significantly higher than customer average")
            if transaction['previous_transactions'] < 10:
                reasons.append("Limited transaction history")
            if transaction['account_age_days'] < 60:
                reasons.append("New account with limited history")
        
        else:
            reasons.append("Normal transaction pattern")
        
        return "; ".join(reasons) if reasons else "No suspicious patterns detected"
    
    def get_checked_transactions(self, limit=50):
        """Get recently checked transactions"""
        return list(self.checked_transactions)[-limit:]
    
    def get_stats(self):
        """Get system statistics"""
        checked_list = list(self.checked_transactions)
        if not checked_list:
            return {
                'total_checked': 0,
                'fraud_count': 0,
                'safe_count': 0,
                'fraud_rate': '0%'
            }
        
        fraud_count = sum(1 for t in checked_list if t.get('is_fraud', False))
        safe_count = len(checked_list) - fraud_count
        
        return {
            'total_checked': len(checked_list),
            'fraud_count': fraud_count,
            'safe_count': safe_count,
            'fraud_rate': f"{(fraud_count/len(checked_list))*100:.1f}%",
            'pending_count': len(self.pending_transactions)
        }
    
    def create_initial_fraud_transactions(self):
        """Create some initial fraudulent transactions for demo"""
        fraud_transactions = [
            {
                'transaction_id': 'TRX-2025-1001',
                'amount': 2850.00,
                'transaction_hour': 2,
                'transaction_day': 3,
                'category': 'Electronics',
                'location': 'Tokyo, JP',
                'device_type': 'Mobile App',
                'previous_transactions': 3,
                'avg_transaction_value': 150.00,
                'customer_age': 25,
                'account_age_days': 15,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                'transaction_id': 'TRX-2025-1002', 
                'amount': 2200.00,
                'transaction_hour': 4,
                'transaction_day': 1,
                'category': 'Travel',
                'location': 'Dubai, AE',
                'device_type': 'Desktop Web',
                'previous_transactions': 8,
                'avg_transaction_value': 200.00,
                'customer_age': 32,
                'account_age_days': 45,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
        
        safe_transactions = [
            {
                'transaction_id': 'TRX-2025-2001',
                'amount': 85.50,
                'transaction_hour': 14,
                'transaction_day': 2,
                'category': 'Food & Dining',
                'location': 'New York, US',
                'device_type': 'Mobile App',
                'previous_transactions': 45,
                'avg_transaction_value': 75.00,
                'customer_age': 40,
                'account_age_days': 800,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
        
        for transaction in fraud_transactions + safe_transactions:
            self.add_pending_transaction(transaction)
        
        print("Created initial transactions for demo")

# Initialize the model
fraud_model = RealTimeFraudDetectionModel()

# Background thread for generating transactions
background_thread_running = True

def generate_transactions_background():
    """Background thread to generate random transactions with 1-minute interval"""
    global background_thread_running
    while background_thread_running:
        try:
            num_transactions = random.randint(1, 3)
            
            for _ in range(num_transactions):
                transaction = fraud_model.generate_random_transaction()
                
                if random.random() < 0.3:
                    transaction['amount'] = transaction['amount'] * random.uniform(2, 5)
                    transaction['transaction_hour'] = random.choice([0, 1, 2, 3, 4, 5])
                    if random.random() < 0.7:
                        transaction['location'] = random.choice([
                            'Tokyo, JP', 'Singapore, SG', 'Dubai, AE', 'Mumbai, IN', 
                            'Online', 'International', 'Hong Kong, HK'
                        ])
                    if random.random() < 0.6:
                        transaction['previous_transactions'] = random.randint(1, 8)
                
                fraud_model.add_pending_transaction(transaction)
                print(f"Generated transaction: {transaction['transaction_id']} - Amount: ${transaction['amount']}")
            
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in background transaction generation: {e}")
            time.sleep(60)

# Start background thread
background_thread = threading.Thread(target=generate_transactions_background, daemon=True)
background_thread.start()

# Authentication Routes
@app.route('/auth/register', methods=['POST'])
@admin_required
def register():
    """Admin only: Register new users"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not all([username, email, password]):
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        c.execute('''
            INSERT INTO users (username, email, password_hash, role, created_by) 
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, role, session['username']))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'User {username} created successfully with role: {role}'
        })
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'message': 'Username or email already exists'}), 400

@app.route('/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and user[4] and check_password_hash(user[2], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        session['role'] = user[3]
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user[0],
                'username': user[1],
                'role': user[3]
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials or inactive account'}), 401

@app.route('/auth/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/auth/users', methods=['GET'])
@admin_required
def get_users():
    """Admin only: Get all users"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, username, email, role, is_active, created_by, created_at 
        FROM users ORDER BY created_at DESC
    ''')
    users = c.fetchall()
    conn.close()
    
    users_list = []
    for user in users:
        users_list.append({
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'role': user[3],
            'is_active': bool(user[4]),
            'created_by': user[5],
            'created_at': user[6]
        })
    
    return jsonify({'success': True, 'users': users_list})

@app.route('/auth/user_status', methods=['PUT'])
@admin_required
def update_user_status():
    """Admin only: Activate/deactivate users"""
    data = request.json
    user_id = data.get('user_id')
    is_active = data.get('is_active')
    
    if user_id == session['user_id']:
        return jsonify({'success': False, 'message': 'Cannot modify your own status'}), 400
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET is_active = ? WHERE id = ?', (is_active, user_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User status updated successfully'})

@app.route('/auth/current_user', methods=['GET'])
def get_current_user():
    """Get current user info"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    return jsonify({
        'success': True,
        'user': {
            'id': session['user_id'],
            'username': session['username'],
            'role': session['role']
        }
    })

# Application Routes - FIXED ROUTES
@app.route('/')
def index():
    return redirect('/login')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('index.html')

# API Routes
@app.route('/api/train_model', methods=['POST'])
@login_required
def train_model():
    """Train the Random Forest model"""
    try:
        results = fraud_model.train_model()
        
        return jsonify({
            'success': True,
            'message': f'Model trained successfully on {results["total_samples"]} transactions!',
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/api/generate_transaction', methods=['POST'])
@login_required
def generate_transaction():
    """Generate a random transaction and add to pending"""
    try:
        transaction = fraud_model.generate_random_transaction()
        pending_count = fraud_model.add_pending_transaction(transaction)
        
        return jsonify({
            'success': True,
            'transaction': transaction,
            'pending_count': pending_count,
            'message': f'Transaction {transaction["transaction_id"]} generated and added to pending'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating transaction: {str(e)}'
        })

@app.route('/api/get_pending_transactions', methods=['GET'])
@login_required
def get_pending_transactions():
    """Get all pending transactions"""
    return jsonify({
        'success': True,
        'transactions': fraud_model.get_pending_transactions(),
        'count': len(fraud_model.pending_transactions)
    })

@app.route('/api/check_transactions', methods=['POST'])
@login_required
def check_transactions():
    """Check all pending transactions for fraud"""
    try:
        results = fraud_model.check_pending_transactions()
        
        return jsonify({
            'success': True,
            'checked_count': len(results),
            'transactions': results,
            'message': f'Checked {len(results)} transactions for fraud'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error checking transactions: {str(e)}'
        })

@app.route('/api/checked_transactions')
@login_required
def get_checked_transactions():
    """Get all checked transactions"""
    limit = request.args.get('limit', 50, type=int)
    transactions = fraud_model.get_checked_transactions(limit)
    
    return jsonify({
        'success': True,
        'transactions': transactions
    })

@app.route('/api/system_stats')
@login_required
def get_system_stats():
    """Get system statistics"""
    stats = fraud_model.get_stats()
    stats['model_trained'] = fraud_model.is_trained
    
    return jsonify({
        'success': True,
        'stats': stats
    })

@app.route('/api/download_report')
@login_required
def download_report():
    """Download transaction report as CSV"""
    try:
        transactions = fraud_model.get_checked_transactions(limit=1000)
        df = pd.DataFrame(transactions)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'fraud_detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating report: {str(e)}'
        })

@app.route('/api/model_status')
@login_required
def model_status():
    """Check if model is trained"""
    return jsonify({
        'is_trained': fraud_model.is_trained,
        'feature_columns': fraud_model.feature_columns if fraud_model.is_trained else [],
        'message': 'Model is ready' if fraud_model.is_trained else 'Model needs training'
    })

def stop_background_thread():
    """Stop the background thread"""
    global background_thread_running
    background_thread_running = False

if __name__ == '__main__':
    try:
        print("Training model with generated data...")
        fraud_model.train_model()
        print("Model trained successfully!")
        
        fraud_model.create_initial_fraud_transactions()
        
        print("Background transaction generation started (1-minute intervals)...")
        print("System is ready!")
        print("Default login: admin/admin123 or analyst/user123")
            
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print("Starting with untrained model. Please train manually via the interface.")
    
    try:
        port = int(os.environ.get("PORT", 5000))  # <-- use Railway's port if available
        app.run(debug=True, host='0.0.0.0', port=port)
    finally:
        stop_background_thread()
