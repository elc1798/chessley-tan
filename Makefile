run: chessley-venv/bin/activate
	source chessley-venv/bin/activate
	python app.py

setup:
	pwd
	echo "Building Chessley Python Virtual Environment"
	sudo pip install --upgrade virtualenv
	virtualenv chessley-venv
	echo "Installing required Python packages"
	sh pip_install.sh
