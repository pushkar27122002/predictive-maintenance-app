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

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load model
# model = joblib.load("multi_rf_model.pkl")

# st.set_page_config(page_title="Predictive Maintenance", layout="centered")

# st.markdown(
#     "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'> Predictive Maintenance for Air Compressor</h1>",
#     unsafe_allow_html=True
# )

# st.markdown(
#     "This app predicts the maintenance condition of:\n"
#     "- Bearings\n"
#     "- Water Pump\n"
#     "- Radiator\n"
#     "- Exhaust Valve"
# )

# st.write("---")

# # Input method
# input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Manually"])

# # Full names for each input
# feature_mapping = {
#     'rpm': "Motor RPM",
#     'motor_power': "Motor Power (kW)",
#     'torque': "Torque (Nm)",
#     'outlet_pressure_bar': "Compressor Outlet Pressure (bar)",
#     'air_flow': "Air Flow Rate (m³/min)",
#     'noise_db': "Noise Level (dB)",
#     'outlet_temp': "Outlet Temperature (°C)",
#     'wpump_outlet_press': "Water Pump Outlet Pressure (bar)",
#     'water_inlet_temp': "Water Inlet Temperature (°C)",
#     'water_outlet_temp': "Water Outlet Temperature (°C)",
#     'wpump_power': "Water Pump Power (kW)",
#     'water_flow': "Water Flow Rate (L/min)",
#     'oilpump_power': "Oil Pump Power (kW)",
#     'oil_tank_temp': "Oil Tank Temperature (°C)",
#     'gaccx': "Gyro Acceleration X",
#     'gaccy': "Gyro Acceleration Y",
#     'gaccz': "Gyro Acceleration Z",
#     'haccx': "Housing Acceleration X",
#     'haccy': "Housing Acceleration Y",
#     'haccz': "Housing Acceleration Z"
# }

# features = list(feature_mapping.keys())

# if input_method == "Upload CSV File":
#     st.markdown("📎 Upload a **CSV file** with the following columns (in any order):")
#     st.code(", ".join(feature_mapping.keys()), language="text")

#     uploaded_file = st.file_uploader("Upload CSV", type="csv")
#     if uploaded_file is not None:
#         input_df = pd.read_csv(uploaded_file)
#         st.subheader("Preview of Uploaded Data")
#         st.dataframe(input_df)

#         if st.button("Predict"):
#             predictions = model.predict(input_df)
#             results = pd.DataFrame(predictions, columns=['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve'])
#             st.success("Prediction completed.")
#             st.dataframe(results)

# else:
#     st.subheader(" Enter Input Values")
#     inputs = []
#     for key in features:
#         value = st.number_input(f"{feature_mapping[key]}", value=0.0)
#         inputs.append(value)

#     if st.button("Predict"):
#         input_array = np.array(inputs).reshape(1, -1)
#         pred = model.predict(input_array)[0]
#         component_names = ['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve']
#         st.markdown("###  Prediction Results")
#         for name, value in zip(component_names, pred):
#             status = "Needs Attention" if value == 1 else "OK"
#             st.write(f"- **{name}**: {status}")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# import io
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas

# # Load model
# model = joblib.load("multi_rf_model.pkl")

# st.set_page_config(page_title="Predictive Maintenance", layout="centered")

# st.markdown(
#     "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'> Predictive Maintenance for Air Compressor</h1>",
#     unsafe_allow_html=True
# )

# st.markdown(
#     "This app predicts the maintenance condition of:\n"
#     "- Bearings\n"
#     "- Water Pump\n"
#     "- Radiator\n"
#     "- Exhaust Valve"
# )

# st.write("---")

# # Input method
# input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Manually"])

# # Full names for each input
# feature_mapping = {
#     'rpm': "Motor RPM",
#     'motor_power': "Motor Power (kW)",
#     'torque': "Torque (Nm)",
#     'outlet_pressure_bar': "Compressor Outlet Pressure (bar)",
#     'air_flow': "Air Flow Rate (m³/min)",
#     'noise_db': "Noise Level (dB)",
#     'outlet_temp': "Outlet Temperature (°C)",
#     'wpump_outlet_press': "Water Pump Outlet Pressure (bar)",
#     'water_inlet_temp': "Water Inlet Temperature (°C)",
#     'water_outlet_temp': "Water Outlet Temperature (°C)",
#     'wpump_power': "Water Pump Power (kW)",
#     'water_flow': "Water Flow Rate (L/min)",
#     'oilpump_power': "Oil Pump Power (kW)",
#     'oil_tank_temp': "Oil Tank Temperature (°C)",
#     'gaccx': "Gyro Acceleration X",
#     'gaccy': "Gyro Acceleration Y",
#     'gaccz': "Gyro Acceleration Z",
#     'haccx': "Housing Acceleration X",
#     'haccy': "Housing Acceleration Y",
#     'haccz': "Housing Acceleration Z"
# }

# features = list(feature_mapping.keys())
# input_df = None

# if input_method == "Upload CSV File":
#     st.markdown("📎 Upload a **CSV file** with the following columns (in any order):")
#     st.code(", ".join(feature_mapping.keys()), language="text")

#     uploaded_file = st.file_uploader("Upload CSV", type="csv")
#     if uploaded_file is not None:
#         input_df = pd.read_csv(uploaded_file)
#         st.subheader("Preview of Uploaded Data")
#         st.dataframe(input_df)

#         if st.button("Predict"):
#             predictions = model.predict(input_df)
#             results = pd.DataFrame(predictions, columns=['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve'])
#             st.success("Prediction completed.")
#             st.dataframe(results)

#         # EDA Section
#         st.subheader("📊 Exploratory Data Analysis")

#         if st.checkbox("Show basic statistics"):
#             st.write(input_df.describe())

#         if st.checkbox("Show correlation heatmap"):
#             fig, ax = plt.subplots(figsize=(12, 10))
#             sns.heatmap(input_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#             st.pyplot(fig)

#         # Generate PDF report
#         if st.button("📥 Download PDF Report"):
#             buffer = io.BytesIO()
#             c = canvas.Canvas(buffer, pagesize=letter)
#             width, height = letter
#             text_obj = c.beginText(40, height - 40)
#             text_obj.setFont("Helvetica", 10)
#             text_obj.textLine("Predictive Maintenance Data Report")
#             text_obj.textLine("")

#             # Add summary stats
#             desc = input_df.describe().round(2)
#             for col in desc.columns:
#                 text_obj.textLine(f"--- {col} ---")
#                 for stat in desc.index:
#                     text_obj.textLine(f"{stat}: {desc.loc[stat, col]}")
#                 text_obj.textLine("")

#             c.drawText(text_obj)
#             c.showPage()
#             c.save()

#             buffer.seek(0)
#             st.download_button(
#                 label="Download Report PDF",
#                 data=buffer,
#                 file_name="predictive_maintenance_report.pdf",
#                 mime="application/pdf"
#             )

# else:
#     st.subheader(" Enter Input Values")
#     inputs = []
#     for key in features:
#         value = st.number_input(f"{feature_mapping[key]}", value=0.0)
#         inputs.append(value)

#     if st.button("Predict"):
#         input_array = np.array(inputs).reshape(1, -1)
#         pred = model.predict(input_array)[0]
#         component_names = ['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve']
#         st.markdown("###  Prediction Results")
#         for name, value in zip(component_names, pred):
#             status = "Needs Attention" if value == 1 else "OK"
#             st.write(f"- **{name}**: {status}")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# import io
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from math import pi
# import plotly.express as px

# # Load model
# model = joblib.load("multi_rf_model.pkl")

# st.set_page_config(page_title="Predictive Maintenance", layout="centered")

# st.markdown(
#     "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'> Predictive Maintenance for Air Compressor</h1>",
#     unsafe_allow_html=True
# )

# st.markdown(
#     "This app predicts the maintenance condition of:\n"
#     "- Bearings\n"
#     "- Water Pump\n"
#     "- Radiator\n"
#     "- Exhaust Valve"
# )

# st.write("---")

# # Input method
# input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Manually", "Single Row"])

# # Full names for each input
# feature_mapping = {
#     'rpm': "Motor RPM",
#     'motor_power': "Motor Power (kW)",
#     'torque': "Torque (Nm)",
#     'outlet_pressure_bar': "Compressor Outlet Pressure (bar)",
#     'air_flow': "Air Flow Rate (m³/min)",
#     'noise_db': "Noise Level (dB)",
#     'outlet_temp': "Outlet Temperature (°C)",
#     'wpump_outlet_press': "Water Pump Outlet Pressure (bar)",
#     'water_inlet_temp': "Water Inlet Temperature (°C)",
#     'water_outlet_temp': "Water Outlet Temperature (°C)",
#     'wpump_power': "Water Pump Power (kW)",
#     'water_flow': "Water Flow Rate (L/min)",
#     'oilpump_power': "Oil Pump Power (kW)",
#     'oil_tank_temp': "Oil Tank Temperature (°C)",
#     'gaccx': "Gyro Acceleration X",
#     'gaccy': "Gyro Acceleration Y",
#     'gaccz': "Gyro Acceleration Z",
#     'haccx': "Housing Acceleration X",
#     'haccy': "Housing Acceleration Y",
#     'haccz': "Housing Acceleration Z"
# }

# features = list(feature_mapping.keys())
# input_df = None

# if input_method == "Upload CSV File":
#     st.markdown("📎 Upload a **CSV file** with the following columns (in any order):")
#     st.code(", ".join(feature_mapping.keys()), language="text")

#     uploaded_file = st.file_uploader("Upload CSV", type="csv")
#     if uploaded_file is not None:
#         input_df = pd.read_csv(uploaded_file)
#         st.subheader("Preview of Uploaded Data")
#         st.dataframe(input_df)

#         if st.button("Predict"):
#             predictions = model.predict(input_df)
#             results = pd.DataFrame(predictions, columns=['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve'])
#             st.success("Prediction completed.")
#             st.dataframe(results)

#         # EDA Section
#         st.subheader("📊 Exploratory Data Analysis")

#         if st.checkbox("Show basic statistics"):
#             st.write(input_df.describe())

#         if st.checkbox("Show correlation heatmap"):
#             fig, ax = plt.subplots(figsize=(12, 10))
#             sns.heatmap(input_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#             st.pyplot(fig)

#         # Generate PDF report
#         if st.button("📥 Download PDF Report"):
#             buffer = io.BytesIO()
#             c = canvas.Canvas(buffer, pagesize=letter)
#             width, height = letter
#             text_obj = c.beginText(40, height - 40)
#             text_obj.setFont("Helvetica", 10)
#             text_obj.textLine("Predictive Maintenance Data Report")
#             text_obj.textLine("")

#             # Add summary stats
#             desc = input_df.describe().round(2)
#             for col in desc.columns:
#                 text_obj.textLine(f"--- {col} ---")
#                 for stat in desc.index:
#                     text_obj.textLine(f"{stat}: {desc.loc[stat, col]}")
#                 text_obj.textLine("")

#             c.drawText(text_obj)
#             c.showPage()
#             c.save()

#             buffer.seek(0)
#             st.download_button(
#                 label="Download Report PDF",
#                 data=buffer,
#                 file_name="predictive_maintenance_report.pdf",
#                 mime="application/pdf"
#             )

# elif input_method == "Single Row":
#     st.subheader("📈 Single Row Input Analysis")

#     row_data = {}
#     for feature in features:
#         row_data[feature] = st.number_input(f"{feature_mapping[feature]}", value=0.0)

#     # Displaying the Radar Chart for the single row
#     if st.button("Generate Radar Chart"):
#         row_df = pd.DataFrame([row_data])

#         # Radar Chart Preparation
#         categories = list(row_data.keys())
#         values = list(row_data.values())

#         # Create the radar chart
#         df_radar = pd.DataFrame([values], columns=categories)
#         df_radar = pd.concat([df_radar, df_radar], axis=0)
#         df_radar.iloc[1] = values

#         # Set up the plot
#         fig, ax = plt.subplots(figsize=(6, 6), dpi=80, subplot_kw=dict(polar=True))
#         angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
#         values += values[:1]
#         angles += angles[:1]

#         ax.fill(angles, values, color='blue', alpha=0.25)
#         ax.plot(angles, values, color='blue', linewidth=2)
#         ax.set_yticklabels([])
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(categories, fontweight='bold')

#         st.pyplot(fig)

#     # Predict and show results
#     if st.button("Predict"):
#         input_array = np.array(list(row_data.values())).reshape(1, -1)
#         pred = model.predict(input_array)[0]
#         component_names = ['Bearings', 'Water Pump', 'Radiator', 'Exhaust Valve']
#         st.markdown("### Prediction Results")
#         for name, value in zip(component_names, pred):
#             status = "Needs Attention" if value == 1 else "OK"
#             st.write(f"- **{name}**: {status}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
input_df = None

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
            results = results.replace({0: "Good Condition", 1: "Needs Attention"})
            st.success("Prediction completed.")
            st.dataframe(results)

        # EDA Section
        st.subheader("📊 Exploratory Data Analysis")

        if st.checkbox("Show basic statistics"):
            st.write(input_df.describe())

        if st.checkbox("Show feature distributions"):
            for column in input_df.columns:
                fig = px.histogram(input_df, x=column, title=f"Distribution of {column}", nbins=30)
                st.plotly_chart(fig)

        if st.checkbox("Show correlation heatmap"):
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(input_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

elif input_method == "Enter Manually":
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
            status = "Needs Attention" if value == 1 else "Works Fine"
            st.write(f"- **{name}**: {status}")
