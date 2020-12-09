from flask import render_template, url_for,request,flash,redirect
import pandas as pd
import numpy as np
import sklearn
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
from flask_main.forms import RegistrationForm, LoginForm, ContactForm
from flask_main import app, db, bcrypt,mail
from flask_main.models import User, Contacts
from flask_login import login_user, current_user ,logout_user ,login_required
from flask_mail import Message

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html',title='Home')

@app.route("/about")
def about():
    return render_template('about.html',title='about')

@app.route("/heart")
@login_required
def heart():
    return render_template('heart.html',title='heart')


@app.route("/diabetes")
@login_required
def diabetes():
    return render_template('diabetes.html',title='diabetes')

@app.route("/cancer")
@login_required
def cancer():
    return render_template('cancer.html',title='cancer')

@app.route("/liver")
@login_required
def liver():
    return render_template('liver.html',title='Liver')

@app.route("/kidney")
@login_required
def kidney():
    return render_template('kidney.html',title='kidney')

@app.route("/mainfile")
def mainfile():
    return render_template('mainfile.html',title='Main')

@app.route("/information")
def information():
    return render_template('information.html',title='Information')


@app.route("/register", methods=['GET', 'POST'])
def registration():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        db.create_all()
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('registration.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page= request.args.get('next')

            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout", methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone_no = request.form.get('phone_no')
        message = request.form.get('message')
        db.create_all()
        entry=Contacts(name=name,email=email,phone_num=phone_no,date=datetime.now(),msg=message)
        db.session.add(entry)
        db.session.commit()
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('contact.html', form=form)
        else:
            msg = Message(subject=f"Mail from {name}",
                          body=f"Name: {name}\nE-mail: {email}\nPhone-no: {phone_no}\n\n\n{message}" ,
                          sender=email, recipients=['m8605199144@gmail.com'])
            'msg.body = """From: %s <%s>%s""" % (name, email, message)'
            mail.send(msg)

            return render_template('contact.html', success=True)

    elif request.method == 'GET':
        return render_template('contact.html', form=form)


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(-1, size)
    if (size == 8):  # Diabetes
        loaded_model = joblib.load("flask_main/models/diabetes_model")
        result = loaded_model.predict(to_predict)
    elif (size == 12):  # Cancer
        loaded_model = joblib.load("flask_main/models/cancer_model")
        result = loaded_model.predict(to_predict)
    # elif(size==12):#Kidney
    #   loaded_model = pickle.load("model3")
    #  result = loaded_model.predict(to_predict)
    elif (size == 7):  # liver
        loaded_model = joblib.load('flask_main/models/liver_modal')
        result = loaded_model.predict(to_predict)
    elif (size == 13):  # Heart
        loaded_model = joblib.load("flask_main/models/heart_model")
        result = loaded_model.predict(to_predict)
    return result[0]

standard_to = StandardScaler()

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if (len(to_predict_list) == 12):  # Cancer
            result = ValuePredictor(to_predict_list, 12)
            if (int(result) == 0):
                prediction = "Congrulation! You DON'T have Cancer disease."
            else:
                prediction = "Oops! You have Cancer disease."
            return (render_template("result.html", prediction=prediction))
        elif (len(to_predict_list) == 8):  # Diabetes
            result = ValuePredictor(to_predict_list, 8)
            if (int(result) == 0):
                prediction = "Congrulation! You DON'T have Diabetes disease."
            else:
                prediction = "Oops! You have Diabetes disease."
            return (render_template("result.html", prediction=prediction))
        # elif(len(to_predict_list)==12):
        #  result = ValuePredictor(to_predict_list,12)
        elif (len(to_predict_list) == 13):  # heart
            result = ValuePredictor(to_predict_list, 13)
            if (int(result) == 0):
                prediction = "Congrulation! You DON'T have Heart disease."
            else:
                prediction = "Oops! You have Heart disease."

            return (render_template("result.html", prediction=prediction))
        elif (len(to_predict_list) == 7):  # Liver
            result = ValuePredictor(to_predict_list, 7)
            if (int(result) == 0):
                prediction = "Congrulation! You DON'T have liver disease."
            else:
                prediction = "Oops! You have liver disease."
            return (render_template("result.html", prediction=prediction))

