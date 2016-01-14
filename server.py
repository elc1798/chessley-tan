from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
@app.route("/home", methods=["GET","POST"])
def home():
    return render_template("home.html")

@app.route("/leaderboards", methods=["GET", "POST"])
def leaderboards():
    return 0
@app.route("/profile", methods=["GET","POST"])
def profile():
    return 0


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=8000)
