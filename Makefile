run: chessley-venv/bin/activate
	source chessley-venv/bin/activate
	python app.py

setup:
	echo "Installing required packages for Ubuntu 14.04"
	./SETUP_FILES/apt_install.sh
	pwd
	echo "Building Chessley Python Virtual Environment"
	sudo pip install --upgrade virtualenv
	virtualenv chessley-venv
	echo "Installing required Python packages"
	./SETUP_FILES/pip_install.sh
	echo "Installing Git LFS"
	./lfs/install.sh
	git lfs pull

update:
	git pull --ff-only --rebase
	git lfs update
	git lfs pull

