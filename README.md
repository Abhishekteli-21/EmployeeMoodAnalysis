
Step 1: Clone the Repository
Open a command prompt (or PowerShell) and navigate to a directory where you want to set up the project:
cd C:\Users\Projects  # Replace with your desired directory
git clone https://github.com/Abhishekteli-21/EmployeeMoodAnalysis.git
cd EmployeeMoodAnalysis



Step 2: Create and Activate a Virtual Environment
Set up a Python virtual environment to isolate dependencies
python -m venv venv
.\venv\Scripts\activate
After activation, your prompt should show (venv) at the start, indicating the virtual environment is active.


Step 3: Install Dependencies
Install the project’s dependencies listed in requirements.txt
pip install -r requirements.txt

step 4: set up the mango db (database)

Step 5: Run the Project
streamlit run app.py
