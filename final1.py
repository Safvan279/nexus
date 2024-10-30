#test with db

import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Route for the homepage
@app.route('/')
def intro():
    return render_template('intro.html')

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))

        # Hash the password and save the new user to the database
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

# Route for the home page after login
@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html', username=session['username'])
    else:
        flash('You need to login first.', 'error')
        return redirect(url_for('login'))

# Route for logging out
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Additional routes and functionalities of your project (existing code)

# Route for displaying the dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return render_template('dashboard.html')  # Add your logic for displaying dashboard data
    else:
        flash('You need to login first.', 'error')
        return redirect(url_for('login'))

# Route for processing the uploaded dataset and displaying analysis results
@app.route('/analyze', methods=['POST'])
def analyze_data():
    # Add your data analysis logic here
    flash('Data analysis complete!', 'success')
    return redirect(url_for('dashboard'))

# If you have other routes, add them below:
# Example of another route:
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
