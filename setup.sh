sudo apt-get update -y
sudo apt-get install python3-venv -y
python3 -m venv env
source ./env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
deactivate