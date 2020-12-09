from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail

app = Flask(__name__,template_folder='template', static_folder='static')


app.config['SECRET_KEY'] = '8b4d5eb374cde67cae8b9dfa477994d3'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] =   465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config["MAIL_USERNAME"] = 'm8605199144@gmail.com'
app.config["MAIL_PASSWORD"] = 'rahulbale143.com'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail=Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

from flask_main import routers