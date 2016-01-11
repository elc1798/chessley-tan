from flask import Flask

app = Flask(__name__)

@app.route("/home", methods=["GET","POST"])
def home():

@app.route("/leaderboards", methods=["GET", "POST"])
def leaderboards():

@app.route("/profile", methods=["GET","POST"])
def profile():



if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=8000)
