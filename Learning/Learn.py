import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


TRAIN_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salary_data.xlsx")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_output.xlsx")
 
# Auto-create salary_data.xlsx if it doesn't exist
if not os.path.exists(TRAIN_FILE):
    sample = pd.DataFrame({
        "Experience": [1, 3, 4, 6, 8],
        "Salary":     [20, 40, 50, 70, 90]
    })
    sample.to_excel(TRAIN_FILE, index=False)
    print(f"Created sample training file: {TRAIN_FILE}\n")
 
 
class LinearRegressionAgent:
    def __init__(self):
        self.model    = LinearRegression()
        self.trained  = False
        self.history  = []
        self.train_df = None
 
    def train_from_excel(self, file_path):
        df = pd.read_excel(file_path)
        self.train_df = df
        X = df[["Experience"]].values
        y = df["Salary"].values
        self.model.fit(X, y)
        self.trained = True
        print(" AI Agent Trained Successfully\n")
        print(f" Using data from: {file_path}\n")
 
    def predict(self, X_new):
        if not self.trained:
            raise Exception(" Train the agent first!")
        return self.model.predict(X_new)
 
    def interact(self):
        if not self.trained:
            print(" Train the agent before interaction.")
            return
 
        print(" Linear Regression AI Agent ready! Type 'stop' to quit.\n")
 
        while True:
            user_input = input("Enter years of experience: ")
            if user_input.lower() == "stop":
                self.export_to_excel()
                self.visualize_scatter()
                print(" Output saved to prediction_output.xlsx")
                print(" Data Analytics chart displayed.")
                break
            try:
                years = float(user_input)
                if years < 0:
                    print(" Invalid input: Years of experience cannot be negative.")
                    continue
                prediction = self.predict([[years]])[0]
                print(f" Predicted Salary for {years} years = ₹{prediction:.2f}k")
                self.history.append([years, prediction])
            except ValueError:
                print(" Enter a valid numeric value.")
 
    # def export_to_excel(self):
    #     if not self.history:
    #         print("No predictions to export.")
    #         return
    #     df = pd.DataFrame(self.history, columns=["Experience", "Predicted Salary"])
    #     df.to_excel(OUTPUT_FILE, index=False)
    #     print(f"\n Predictions saved to: {OUTPUT_FILE}\n")
 
    def visualize_scatter(self):
        df     = pd.read_excel(OUTPUT_FILE)
        X      = df[["Experience"]].values
        y      = df["Predicted Salary"].values
        model  = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
 
        sort_idx      = X.flatten().argsort()
        X_sorted      = X[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
 
        plt.figure()
        plt.scatter(X, y, color="green", label="Predicted Data")
        plt.plot(X_sorted, y_pred_sorted, color="red", label="Regression Line")
        plt.xlabel("Experience (years)")
        plt.ylabel("Predicted Salary (₹k)")
        plt.title("Predicted Salary vs Experience")
        plt.legend()
        plt.show()



agent = LinearRegressionAgent()
agent.train_from_excel(TRAIN_FILE)
agent.interact()