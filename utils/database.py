import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brahamai.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users Table (Stores regular users and admins, along with remaining credits)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            is_admin BOOLEAN DEFAULT 0,
            credits INTEGER DEFAULT 5
        )
    ''')
    
    # Search History Logger
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            budget INTEGER,
            travel_type TEXT,
            season TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Generate default Admin account if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        create_user("admin", "admin123", is_admin=True, credits=9999)
        
    conn.commit()
    conn.close()

def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, is_admin=False, credits=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = _hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password_hash, is_admin, credits) VALUES (?, ?, ?, ?)",
                       (username, pwd_hash, is_admin, credits))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False # Username already exists
    conn.close()
    return success

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = _hash_password(password)
    
    cursor.execute("SELECT is_admin, credits FROM users WHERE username=? AND password_hash=?", (username, pwd_hash))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"username": username, "is_admin": bool(user[0]), "credits": user[1]}
    return None

def get_credits(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT credits FROM users WHERE username=?", (username,))
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else 0

def use_credit(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Check current credits first
    cursor.execute("SELECT credits FROM users WHERE username=?", (username,))
    res = cursor.fetchone()
    if res and res[0] > 0:
        cursor.execute("UPDATE users SET credits = credits - 1 WHERE username=?", (username,))
        conn.commit()
        success = True
    else:
        success = False
    conn.close()
    return success

def add_credits(username, amount):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET credits = credits + ? WHERE username=?", (amount, username))
    conn.commit()
    conn.close()

def log_search(username, budget, travel_type, season):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO search_history (username, budget, travel_type, season) VALUES (?, ?, ?, ?)",
                   (username, budget, travel_type, season))
    conn.commit()
    conn.close()

# Initialize database mapping structure when imported
init_db()
