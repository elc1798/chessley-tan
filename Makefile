run: chessley-venv/bin/activate
	source chessley-venv/bin/activate
	python app.py

setup:
	pwd
	echo "Building Chessley Python Virtual Environment"
	sudo pip install --upgrade virtualenv
	virtualenv chessley-venv
	echo "Installing required Python packages"
	bash pip_install.sh
	echo "Installing Git LFS"
	./lfs/install.sh
	git lfs pull
