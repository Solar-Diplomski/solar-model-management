## Starting the service


Activate the virtual environment assuming you created your venv in this folder:
Windows: ```venv\Scripts\activate```
macOS/Linux: ```source venv/bin/activate```

Install dependencies if you don't have them:
```pip install -r requirements.txt```

Run the application using Uvicorn:
```uvicorn app:app --reload```
