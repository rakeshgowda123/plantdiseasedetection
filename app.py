from flask import Flask ,redirect , url_for , render_template,request,jsonify,flash, session
from flask_mysqldb import MySQL
import datetime
import os
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
# from alovera_leaf_disease import Net
from Net import Net

import base64
from io import BytesIO
from PIL import Image

from flask_cors import CORS


app = Flask(__name__)
CORS(app) 

app.secret_key = 'supersecretkey'  # Change this to a random and secure key in a real application

# Temporary user data storage (replace with a database in a real-world scenario)
users = []


# routes 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
   filename = None
   if request.method == 'POST':
      file = request.files['image']
      if file:
         target_directory = 'static/uploads/'
         new_filename = 'test_image.jpg'
         absolute_path = os.path.join(target_directory, new_filename)
         file.save(absolute_path)

         img_height = 180
         img_width = 180
         
         device = "cuda" if torch.cuda.is_available() else "cpu"
         device

         # Class names and mapping
         class_names = ['healthy_leaf', 'rot', 'rust']
         num_classes = len(class_names) 
         model = Net(num_classes,img_height,img_width).to(device) 
         model.load_state_dict(torch.load('CNN_model123.pth'))
         # {1: 'healthy_leaf', 2: 'rot', 3: 'rust'}

         # Read the dataset
         df = pd.read_csv("model/data_description.csv")
         df = df.replace('',np.nan)
         df = df.dropna(axis="rows",how="all")
         df = df.dropna(axis="columns",how="all")

         path = absolute_path

         directory_path = "static/uploads/"
         file_name = 'test_image.jpg'

         path = os.path.join(directory_path, file_name)

         if os.path.exists(path):
            print(f"The file path is: {path}")
         else:
            print(f"The file {path} does not exist.")

         #something
         # Define transformations and preprocessing
         preprocess = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
         ])
               #  Load and preprocess the image, then remove alpha channel if it exists
         img = Image.open(path).convert('RGB')
         img = preprocess(img)
         img = img.unsqueeze(0)  # Add a batch dimension

               # Make a prediction with your PyTorch model
         model.eval()  # Set the model to evaluation mode
         with torch.no_grad():
                  prediction = model(img)



               # Get the predicted class
         predicted_class = torch.argmax(prediction).item()

         Disease	,Description	,Symptoms	,Diagnosis	,Precaution	,Medicine_for_cure	,Stage	,Severity	,Recommended_treatment	,Remedy_to_cure	 = df.loc[predicted_class,:]


         print(f"Predicted class: {predicted_class}")
         print(f"Predicted disease: {class_names[predicted_class]}")
         print(f"Disease : {Disease}\nDescription : {Description}\nSymptoms : {Symptoms}\nDiagnosis : {Diagnosis}\nPrecaution : {Precaution}\nSeverity : {Severity}\nRecommended_treatment : {Recommended_treatment}\nRemedy_to_cure : {Remedy_to_cure}")


         

         # Extract information from the dataset
         # Disease,Description,Symptoms,Diagnosis,Precaution,Medicine_for_cure,Stage,Severity,Recommended_treatment,Remedy_to_cure = df.loc[predicted_class, :]
         Disease	,Description	,Symptoms	,Diagnosis	,Precaution	,Medicine_for_cure	,Stage	,Severity,	Recommended_treatment	,Remedy_to_cure	 = df.loc[predicted_class, :]
         path = os.path.abspath(path)

         # dataset = {
         #    "Disease_Type": Disease_Type,
         #    "Severity": Severity,
         #    "Description": Description,
         #    "Symptoms": Symptoms,
         #    "Diagnosis": Diagnosis,
         #    "Precautions": Precautions
            # }
      return render_template('predict.html', disease_name=Disease,desc=Description,Symptoms=Symptoms,Recommendedtreatment=Recommended_treatment,Remedy=Remedy_to_cure)

   return render_template('dashboard.html')


@app.route('/')
def index():
   return render_template('index.html')

@app.route('/more')
def more():
   return render_template('more.html')
# @app.route('/login',methods=['GET', 'POST'])
# def login():
#    if request.method == 'POST':
#       username = request.form['username']
#       phone_no = request.form['ph_no']

#       return redirect('/predict')
#    return render_template('login.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user exists and the password is correct
        if any(user['username'] == username and user['password'] == password for user in users):
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')
 


@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html')

# @app.route('/more')
# def more():
#     return render_template('more.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    enable_login_button = False  
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username is already taken
        if any(user['username'] == username for user in users):
            flash('Username is already taken. Please choose another.', 'error')
            enable_login_button = True
            
            
        else:
            users.append({'username': username, 'password': password})
            flash('Account created successfully. You can now log in.', 'success')
            return redirect(url_for('index'))

    return render_template('sign_up.html', enable_login_button=enable_login_button)
   
   
   

if __name__ == '__main__':
   app.run(debug=True)