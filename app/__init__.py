from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'loan-approval-local-dev-key'

from . import routes