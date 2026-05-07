import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# -------------------- LOAD MODEL & ENCODERS --------------------
model = tf.keras.models.load_model('salary_regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -------------------- UI DESIGN --------------------
st.title("💰 Salary Prediction Dashboard")
st.markdown("Predict estimated salary using a trained ANN model")

# -------------------- LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)

    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

with col2:
    st.subheader("Account Details")
    balance = st.number_input('Balance', value=0.0)
    credit_score = st.number_input('Credit Score', value=600)
    tenure = st.slider('Tenure (Years)', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    exited = st.selectbox('Exited Bank', [0, 1])

# -------------------- INPUT PREPARATION --------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encoding geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([
    input_data.reset_index(drop=True),
    geo_encoded_df.reset_index(drop=True)
], axis=1)

# Scaling
input_data_scaled = scaler.transform(input_data)

# -------------------- PREDICTION BUTTON --------------------
st.markdown("---")
if st.button("Predict Salary"):
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    col3, col4, col5 = st.columns(3)

    with col4:
        st.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")

    st.success("Prediction completed successfully!")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with TensorFlow + Streamlit")










# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# ## Load the trained model
# model = tf.keras.models.load_model('salary_regression_model.h5')

# ## Load the encoder and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#   label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#   onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#   scaler = pickle.load(file)

# ## Streamlit App
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# exited = st.selectbox('Exited', [0, 1])    
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4) 
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# ## Prepare input data
# input_data = pd.DataFrame({
#   'CreditScore': [credit_score],
#   'Gender': [label_encoder_gender.transform([gender])[0]],
#   'Age': [age],
#   'Tenure': [tenure],
#   'Balance': [balance],
#   'NumOfProducts': [num_of_products],
#   'HasCrCard': [has_cr_card],
#   'IsActiveMember': [is_active_member],
#   'Exited': [exited]
# })

# ## One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# ## Combine one-hot encoded columns with input data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# ## Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Predict estimated salary
# prediction = model.predict(input_data_scaled)
# predicted_salary = prediction[0][0]

# st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')




