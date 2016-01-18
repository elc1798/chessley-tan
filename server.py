from flask import Flask, render_template, request, session, redirect, url_for
import os

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
@app.route("/home", methods=["GET","POST"])
def home():
    if request.method=="GET":
        if 'username' in session and session['username']!=0:
            return render_template("home.html", username=session['username'])
        return render_template("home.html")
    else:
        if 'username' in session and session['username']!=0:
            return render_template("home.html", username=session['username'])
        return render_template("home.html")

@app.route("/leaderboards", methods=["GET", "POST"])
def leaderboards():
    if request.method=="GET":
        return render_template("leaderboards.html")
    else:
        if 'username' in session and session['username']!=0:
            return render_template("leaderboards.html", username = session['username'])
        return render_template("leaderboards.html")

@app.route("/profile", methods=["GET","POST"])
def profile():
    if 'username' in session and session['username']!=0:
        return render_template("profile.html", username=session['username'])
    return render_template("home.html")


if __name__ == "__main__":
    app.debug = True
    app.secret_key = str(os.urandom(24))
    app.run(host="0.0.0.0", port=8000)
