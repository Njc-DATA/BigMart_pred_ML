import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
try:
    model = joblib.load("Packages/bigmart_model.h5")
except FileNotFoundError:
    st.error("The model file 'bigmart_model.h5' was not found.")
    st.stop()

# Define the ordinal encoder categories. 
# These are the categories derived from your training code.
# The order is crucial as it was used during model training.
# Note: You need to manually define the categories based on the output of your original training script.
# The categories below are an educated guess based on the problem description. 
# You should verify them from your 'print(oe.categories_)' output.
outlet_identifier_categories = np.array(['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
outlet_size_categories = np.array(['High', 'Medium', 'Small'])
outlet_type_categories = np.array(['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

# Define a function to preprocess user input
def preprocess_input(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_age):
    """
    This function applies the same preprocessing steps as the model training pipeline.
    """
    
    # Create a DataFrame from the user input
    input_data = pd.DataFrame([[item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_age]],
                              columns=["Item_MRP", "Outlet_Identifier", "Outlet_Size", "Outlet_Type", "Outlet_age"])

    # Ordinal Encoding for categorical features.
    # We use a dictionary to store the mappings based on the original encoder categories.
    encoding_maps = {
        'Outlet_Identifier': {cat: i for i, cat in enumerate(outlet_identifier_categories)},
        'Outlet_Size': {cat: i for i, cat in enumerate(outlet_size_categories)},
        'Outlet_Type': {cat: i for i, cat in enumerate(outlet_type_categories)}
    }

    # Apply the encoding to the input data
    input_data["Outlet_Identifier"] = input_data["Outlet_Identifier"].map(encoding_maps['Outlet_Identifier'])
    input_data["Outlet_Size"] = input_data["Outlet_Size"].map(encoding_maps['Outlet_Size'])
    input_data["Outlet_Type"] = input_data["Outlet_Type"].map(encoding_maps['Outlet_Type'])
    
    # Ensure all columns are numeric
    input_data = input_data.apply(pd.to_numeric)
    
    return input_data

# Set up the Streamlit app interface
st.title("Big Mart Sales Prediction App 🛒")
st.write("Enter the item and outlet details to predict the sales.")

# Create input widgets for the features
item_mrp = st.number_input("Item MRP (Maximum Retail Price)", min_value=1.0, value=100.0)

# Use selectboxes for categorical features
outlet_identifier = st.selectbox("Outlet Identifier", outlet_identifier_categories)
outlet_size = st.selectbox("Outlet Size", outlet_size_categories)
outlet_type = st.selectbox("Outlet Type", outlet_type_categories)

# An integer slider is great for a numerical feature like age
outlet_age = st.slider("Outlet Age (Years)", min_value=1, max_value=40, value=10)

# Create a prediction button
if st.button("Predict Sales"):
    # Preprocess the user's input
    processed_data = preprocess_input(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_age)
    
    # Make a prediction using the loaded model
    prediction = model.predict(processed_data)
    

# Add an optional section to show the processed input
    st.subheader("Input Data (Processed)")
    st.write(processed_data)

    # Display the result
    st.success(f"Predicted Item Outlet Sales: ${prediction[0]:.2f}")
    
    