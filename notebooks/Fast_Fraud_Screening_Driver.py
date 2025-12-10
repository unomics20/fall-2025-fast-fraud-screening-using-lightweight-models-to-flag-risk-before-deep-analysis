"""
This Fast_Fraud_Screening_Driver_ntbk.py
--------------------------------
This script runs Jupyter notebooks for Fast_Fraud_Screening_Model.ipynb, Fast_Fraud_Screening_Business_KPIs.ipynb, and Fast_Fraud_Screening_Report_Visualisations.ipynb programmatically using nbclient.
Each notebook will execute cell by cell — exactly as if you ran it manually in Jupyter. This notebook will print errors from any notebook however, it will not 
stop the next notebook from running.


#project structure
Fast_Fraud_Screeening_Project/
│
├── Fast_Fraud_Screening_Model.ipynb
├── Fast_Fraud_Screening_Business_KPIs.ipynb
├── Fast_Fraud_Screening_Report_Visualization.ipynb
└── Fast_Fraud_Screening_Driver.py  <- main driver script

Usage:
    Fast_Fraud_Screening_Driver.py
"""

from nbclient import NotebookClient
from nbformat import read
from pathlib import Path
import traceback

def run_notebook(path):
    """
    Executes a Jupyter notebook and prints success or error messages.

    Parameters
    ----------
    path : str or Path
        The path to the notebook file (.ipynb)
    """
    nb_path = Path(path)

    if not nb_path.exists():
        print(f" Notebook not found: {nb_path}")
        return

    print(f"Running notebook: {nb_path.name}")

    try:
        #Read the notebook file
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = read(f, as_version=4)

        #Create a client and execute the notebook
        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute()

        print(f"Finished running: {nb_path.name}\n")

    except Exception as e:
        print(f"Error while running {nb_path.name}: {e}")
        traceback.print_exc()
        print()

def main():
    """
    Runs our sequence of notebooks in order.
    """

    notebooks = [
   "/Fast_Fraud_Screening_Model.ipynb",
   "/Fast_Fraud_Screening_Business_KPIs.ipynb",
   "/Fast_Fraud_Screening_Report_Visualisations.ipynb"
]

    for nb in notebooks:
        run_notebook(nb)

    print("All notebooks executed.")

if __name__ == "__main__":
    main()
