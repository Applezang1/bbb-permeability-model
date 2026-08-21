#!/usr/bin/env python3

""" 
BBB Permeability prediction using a Deep Neural Network 
as published in  https://doi.org/10.3389/fnins.2022.858126

It calculates descriptors using PaDEL, and uses them to
predict BBB permeability with a trained DNN model. 

Compounds' SMILES must be in .smi file in the same folder.
Training data (data.csv), model (DeePredmodel.h5), and PaDEL
folder must be in the same folder, too. 

Edited by Dr. Freddy Bernal 
"""

import os
import sys
import pandas as pd  
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


# Create function to get predictions using trained model
def bbb(input_bbb: pd.DataFrame) -> np.ndarray:
    """Function to predict BBB permeability for a set
    set of samples with PaDEL features.

    Args:
        input_bbb (pd.DataFrame): PaDEL Features for compounds

    Returns:
        np.ndarray: BBB class (1 permeable; 0 non-permeable)
    """
    # Load training data and scale it 
    bbb_data = pd.read_csv(os.path.join(path, "data.csv"), header=None)
    scaler = StandardScaler()  
    bbb_data = scaler.fit_transform(bbb_data)  
    input_bbb = input_bbb.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Transform user data to numpy to avoid conflict with names
    bbb_user_input = scaler.transform(input_bbb.to_numpy())
    # Load model
    loaded_model = load_model(os.path.join(path, "DeePredmodel.h5")) 
    print("Model loaded")
    # Get predictions for user input
    prediction = loaded_model.predict(bbb_user_input).round()
    prediction = prediction[:,0].astype (int)
    
    return prediction
    

# Create main function to run descriptor calculation and predictions 
def run_prediction(folder: str) -> None:
    """Function to calculate descriptors (using PaDEL) and to generate
    predictions of BBB permeability for a set of compounds (SMILES).

    Args:
        folder (str): Folder to search for ".smi" file (multiple structures)

    Returns:
        CSV file with resulting BBB class (1 permeable; 0 non-permeable)
    """
    removed_molecule=0
    # Define command for PaDEL
    padel_cmd = [
        'java', '-jar', 
        os.path.join(path, 'PaDEL-Descriptor/PaDEL-Descriptor.jar'),
        '-descriptortypes', 
        '-threads', '-1',
        os.path.join(path, 'PaDEL-Descriptor/descriptors.xml'), 
        '-dir', folder, '-file', folder + '/PaDEL_features.csv', 
        '-2d', '-fingerprints', '-removesalt', '-detectaromaticity', 
        '-standardizenitro']
    # Calculate features
    try: 
        subprocess.run(padel_cmd, timeout=600, check=True)
    except subprocess.TimeoutExpired:
        print('Feature calculations timed out')
        removed_molecule += 1
        sys.exit(1)
    print("Features calculated")

    # 1. Load the raw features CSV from PaDEL
    raw_df = pd.read_csv(folder + "/PaDEL_features.csv")

    # 2. Replace Infinity / -Infinity with NaN
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)

    # 3. Drop every row that contains AT LEAST ONE NaN value
    clean_df = raw_df.dropna().reset_index(drop=True)

    # Optional: Log how many bad molecules were removed
    removed_count = len(raw_df) - len(clean_df)
    if removed_count > 0:
        print(f"Removed {removed_count} compound(s) due to invalid/NaN descriptors.")

    # 4. Extract the names and descriptor features from the CLEANED DataFrame
    names = clean_df['Name'].copy()
    input_bbb = clean_df.drop(['Name'], axis=1)

    # 5. Run predictions on valid molecules only
    pred = bbb(input_bbb)

    # 6. Save final results
    res = pd.DataFrame({'Name': names, 'Predicted_class': pred})
    res.to_csv('DeePred-BBB_predictions.csv', index=False)

    '''
    # Create Dataframe for calculated features
    input_bbb =pd.read_csv(folder + "/PaDEL_features.csv")
    # Store name of each sample
    names = input_bbb['Name'].copy()
    # Remove names
    input_bbb = input_bbb.drop(['Name'], axis=1)
    # Run predictions
    pred = bbb(input_bbb)
    # Create Dataframe with results
    res = pd.DataFrame(names)
    res['Predicted_class'] = pred
    # Save results to csv
    res.to_csv('DeePred-BBB_predictions.csv', index=False)'''

    return None
    

# Run script
if __name__ == "__main__":
    # Define current directory
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = os.getcwd()
    # Verify existence of file with SMILES
    exists = [fname for fname in os.listdir(path) if fname.endswith(".smi")]
    if exists:
        # Get predictions
        run_prediction(path)        
    else:
        raise FileNotFoundError("Input File NOT found")
    