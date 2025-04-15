# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load trained model
# model = joblib.load("multi_rf_model.pkl")

# st.title("🔧 Predictive Maintenance Web App")

# st.write("This app predicts the condition of 4 components:\n- Bearings\n- Water Pump\n- Radiator\n- Exhaust Valve")

# input_method = st.radio("Choose input method:", ["Upload CSV", "Manual Input"])

# # Features used in training
# feature_names = [
#     'rpm', 'motor_power', 'torque', 'outlet_pressure_bar', 'air_flow',
#     'noise_db', 'outlet_temp', 'wpump_outlet_press', 'water_inlet_temp',
#     'water_outlet_temp', 'wpump_power', 'water_flow', 'oilpump_power',
#     'oil_tank_temp', 'gaccx', 'gaccy', 'gaccz', 'haccx', 'haccy', 'haccz'
# ]

# if input_method == "Upload CSV":
#     file = st.file_uploader("Upload a CSV file", type=["csv"])
#     if file is not None:
#         df_input = pd.read_csv(file)
#         st.write("Data Preview:")
#         st.dataframe(df_input)

#         if st.button("Predict"):
#             preds = model.predict(df_input)
#             pred_df = pd.DataFrame(preds, columns=['bearings', 'wpump', 'radiator', 'exvalve'])
#             st.write("🔍 Prediction Results:")
#             st.dataframe(pred_df)

# else:
#     st.subheader("Enter Input Values:")
#     user_input = []
#     for feature in feature_names:
#         val = st.number_input(f"{feature}", value=0.0)
#         user_input.append(val)

#     if st.button("Predict"):
#         input_array = np.array(user_input).reshape(1, -1)
#         prediction = model.predict(input_array)[0]

#         st.markdown("### 🔧 Maintenance Prediction")
#         results = {
#             'Bearings': prediction[0],
#             'Water Pump': prediction[1],
#             'Radiator': prediction[2],
#             'Exhaust Valve': prediction[3]
#         }
#         for k, v in results.items():
#             st.write(f"{k}: {'Needs Attention' if v else 'OK'}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("multi_rf_model.pkl")

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.markdown(
    "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'> Predictive Maintenance for Air Compressor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "This app predicts the maintenance condition of:\n"
    "- Bearings\n"
    "- Water Pump\n"
    "- Radiator\n"
    "- Exhaust Valve"
)

st.write("---")

# Input method
input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Manually"])

# Full names for each input
feature_mapping = {
    'rpm': "Motor RPM",
    'motor_power': "Motor Power (kW)",
    'torque': "Torque (Nm)",
    'outlet_pressure_bar': "Compressor Outlet Pressure (bar)",
    'air_flow': "Air Flow Rate (m³/min)",
    'noise_db': "Noise Level (dB)",
    'outlet_temp': "Outlet Temperature (°C)",
    'wpump_outlet_press': "Water Pump Outlet Pressure (bar)",
    'water_inlet_temp': "Water Inlet Temperature (°C)",
    'water_outlet_temp': "Water Outlet Temperature (°C)",
    'wpump_power': "Water Pump Power (kW)",
    'water_flow': "Water Flow Rate (L/min)",
    'oilpump_power': "Oil Pump Power (kW)",
    'oil_tank_temp': "Oil Tank Temperature (°C)",
    'gaccx': "Gyro Acceleration X",
    'gaccy': "Gyro Acceleration Y",
    'gaccz': "Gyro Acceleration Z",
    'haccx': "Housing Acceleration X",
    'haccy': "Housing Acceleration Y",
    'haccz': "Housing Acceleration Z"
}

features = list(feature_mapping.keys())

if input_method == "Upload CSV File":
    st.markdown("📎 Upload a **CSV file** with the following columns (in any order):")
    st.code(", ".join(feature_mapping.keys()), language="text")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(input_df)

        if st.button("Predict"):
            predictions = model.predict(input_df)
            results = pd.DataFrame(predictions, columns=['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve'])
            st.success("Prediction completed.")
            st.dataframe(results)

else:
    st.subheader(" Enter Input Values")
    inputs = []
    for key in features:
        value = st.number_input(f"{feature_mapping[key]}", value=0.0)
        inputs.append(value)

    if st.button("Predict"):
        input_array = np.array(inputs).reshape(1, -1)
        pred = model.predict(input_array)[0]
        component_names = ['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve']
        st.markdown("###  Prediction Results")
        for name, value in zip(component_names, pred):
            status = "Needs Attention" if value == 1 else "OK"
            st.write(f"- **{name}**: {status}")

