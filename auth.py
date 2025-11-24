# auth.py
from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

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
    
    conn.commit()
    conn.close()

init_db()

@auth_bp.route('/register', methods=['POST'])
def register():
    """Admin only: Register new users"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    # Check if current user is admin
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    
    if not user or user[0] != 'admin':
        conn.close()
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not all([username, email, password]):
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
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

@auth_bp.route('/login', methods=['POST'])
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

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@auth_bp.route('/users', methods=['GET'])
def get_users():
    """Admin only: Get all users"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    
    if not user or user[0] != 'admin':
        conn.close()
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
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

@auth_bp.route('/user_status', methods=['PUT'])
def update_user_status():
    """Admin only: Activate/deactivate users"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    
    if not user or user[0] != 'admin':
        conn.close()
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    data = request.json
    user_id = data.get('user_id')
    is_active = data.get('is_active')
    
    if user_id == session['user_id']:
        conn.close()
        return jsonify({'success': False, 'message': 'Cannot modify your own status'}), 400
    
    c.execute('UPDATE users SET is_active = ? WHERE id = ?', (is_active, user_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User status updated successfully'})

@auth_bp.route('/current_user', methods=['GET'])
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