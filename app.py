from flask import Flask,render_template ,request
import requests


app = Flask(__name__)


@app.route("/")
def index(request):
    return render_template(request)