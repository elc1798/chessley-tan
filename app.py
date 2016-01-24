import os
import module
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from werkzeug import secure_filename

app = Flask(__name__)

##Configure upload locations
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])

##Checks if uploaded file is a valid file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
@app.route("/home")
def home():
    if 'username' in session:
        return render_template("logintemplate.html")
    return render_template("home.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    else:
        if 'username' in request.form and 'pass' in request.form and module.authenticate(request.form['username'], request.form['pass']):
            session['authenticated'] = True
            session['username'] = request.form['username']
            return redirect(url_for('home'))
        return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    if 'username' in request.form and 'pass' in request.form and 'pass2' in request.form:
        if not request.form["pass"] == request.form["pass2"]:
            return redirect(url_for('home'))
        if module.newUser(request.form["username"], request.form["pass"]):
            session['authenticated'] = True
            session['username'] = request.form['username']
            return redirect(url_for('home'))
    return redirect(url_for('home'))

@app.route("/upload", methods=["GET","POST"])
def upload():
    if 'username' in session and session['username'] !=0:
        if request.method=="GET":
            return render_template("upload.html")
        else:
            file = request.files['logs']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('profile'))
    else:
        return redirect(url_for('home'))

@app.route("/leaderboards", methods=["GET", "POST"])
def leaderboards():
    if request.method=="GET":
        return render_template("leaderboards.html")
    else:
        if 'username' in session and session['username']!=0:
            return render_template("loginleaderboards.html")
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
