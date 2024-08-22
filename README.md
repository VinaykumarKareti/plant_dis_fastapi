

**Required Packages**
python3 -m venv myenv
source myenv/bin/activate
pip install fastapi uvicorn torch torchvision fastapi python-multipart





**To run the application**
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
