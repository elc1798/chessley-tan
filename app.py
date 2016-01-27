import os
import module

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from functools import wraps

app = Flask(__name__)

# Configure upload locations
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['chessley']) # Change this to whatever filetype to accept

# Checks if uploaded file is a valid file
def allowed_file(filename):
    """
    Checks if 'filename' is allowed to be uploaded to the server

    Params:
        filename - String containing the name of the uploaded file

    Returns:
        True if the file is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

# Wraps for login requirements on certain app.routes

def login_required(f):
    """
    Python function wrapper, used on functions that require being logged in to
    view. Run before a function's body is run.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "authenticated" not in session or not session["authenticated"] or \
            "username" not in session:
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def redirect_if_logged_in(f):
    """
    Python function wrapper, used on functions to redirect to other pages if
    the user is already logged in. Run before a function's body is run.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "authenticated" in session and session["authenticated"]:
            return redirect(url_for("profile"))
        return f(*args, **kwargs)
    return decorated_function

############### APPLICATION SITE ROUTES  ###############

@app.route("/")
@app.route("/home")
@app.route("/home/")
@redirect_if_logged_in
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET","POST"])
@app.route("/login/", methods=["GET","POST"])
@redirect_if_logged_in
def login():
    if request.method == "POST":
        REQUIRED = ["username", "pass"]
        for form_elem in REQUIRED:
            if form_elem not in request.form:
                return render_template("login.html")
        if module.authenticate(request.form['username'], request.form['pass']):
            session["authenticated"] = True
            session["username"] = request.form['username']
            return redirect(url_for("profile"))
    return render_template("login.html")

@app.route("/logout")
@app.route("/logout/")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register", methods=["POST"])
@app.route("/register/", methods=["POST"])
@redirect_if_logged_in
def register():
    REQUIRED = ["username", "pass", "pass2"]
    for form_elem in REQUIRED:
        if form_elem not in request.form:
            return redirect(url_for("home"))
    if request.form["pass"] != request.form["pass2"]:
        return redirect(url_for("home"))
    if module.newUser(request.form["username"], request.form["pass"]):
        session['authenticated'] = True
        session['username'] = request.form['username']
        return redirect(url_for("profile"))
    else:
        return redirect(url_for("home"))

@app.route("/about")
@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/download", methods=["GET", "POST"])
@app.route("/download/", methods=["GET", "POST"])
@login_required
def download():
    return render_template('download.html', USERNAME=session['username']) # For when the Jinja is configured

@app.route("/upload", methods=["GET","POST"])
@app.route("/upload/", methods=["GET","POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files["upload_bot"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename + session["username"] + "_bot.chessley"))
    return render_template("upload.html")

@app.route("/leaderboards", methods=["GET", "POST"])
@app.route("/leaderboards/", methods=["GET", "POST"])
def leaderboards():
    table = module.getRankedUsers()
    if 'username' in session and session['username']!=0:
        return render_template("loginleaderboards.html", table=table)
    return render_template("leaderboards.html", table=table)

@app.route("/profile", methods=["GET","POST"])
@app.route("/profile/", methods=["GET","POST"])
@login_required
def profile():
    if 'username' in session and session['username']!=0:
        #retrieve user data here
        dict = module.getUser(session['username'])
        #dict = {"rank":1,"elo":1400,"wins":100,"losses":50,"stalemates":0}
        return render_template("profile.html", USERNAME=session['username'], DICT=dict)
    return render_template("home.html")

if __name__ == "__main__":
    app.debug = True
    app.secret_key = str(os.urandom(24))
    app.run(host="0.0.0.0", port=5000)
