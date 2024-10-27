# Data Loading & Pre-Processing
import pandas as pd
from imblearn.over_sampling import SMOTE
import sys
print("Sys: ", sys.executable)

# Machine Learning
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# ML Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# Data Visualization
import gradio as gr
import plotly.express as px

# Other imports
import os
import warnings
import joblib
from fpdf import FPDF
warnings.filterwarnings('once')
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

# Loading the resampled data directly
X_train_resampled_df = pd.read_csv('smoted_dataset/X_train_resampled.csv')
y_train_resampled_df = pd.read_csv('smoted_dataset/y_train_resampled.csv')

# Converting y_train_resampled_df back to a 1D array (as it may load as a DataFrame with one column)
y_train_resampled = y_train_resampled_df['Diagnosis'].values

# Loading pre-trained models
model_save_path = 'saved_models'
xgboost_model = joblib.load(os.path.join(model_save_path, 'XGBoost.joblib'))
catboost_model = joblib.load(os.path.join(model_save_path, 'CatBoost.joblib'))
decision_tree_model = joblib.load(os.path.join(model_save_path, 'Decision Tree.joblib'))
random_forest_model = joblib.load(os.path.join(model_save_path, 'Random Forest.joblib'))

# Establishing the model pipeline
model_pipeline = [xgboost_model, catboost_model, decision_tree_model, random_forest_model]
for model in model_pipeline:
    print(type(model))  # debugging
model_list = ['XGBoost', 'CatBoost', 'Decision Tree', 'Random Forest']

# Loading the pre-saved scaler (RobustScaler) and the label encoder
scaler = joblib.load('pre-trained scaler/scaler.joblib') 
label_encoder = joblib.load('pre-trained label encoder/label_encoder.joblib')

original_columns = X_train_resampled_df.columns
input_fields = [gr.Number(label = col) for col in original_columns]

diagnosis_options = y_train_resampled_df['Diagnosis'].unique().tolist()
input_fields.append(gr.Dropdown(label = 'Select Diagnosis for Distribution Plot', choices = diagnosis_options, value = 'Healthy', info = 'Select a diagnosis to compare your data to the existing diagnosis distribution.'))

def predict_diagnosis(*args):
    try:
        patient_data = pd.DataFrame([args[:-1]], columns = original_columns) # creating a DataFrame from the user inputs
        patient_data_scaled = scaler.transform(patient_data)            # scaling the input data using the same scaler as used in training
        
        predicted_diagnosis = []
        for i, model in enumerate(model_pipeline):                      # predicting the diagnosis using each model
            patient_pred = model.predict(patient_data_scaled).ravel()
            patient_pred = label_encoder.inverse_transform(patient_pred)
            predicted_diagnosis.append(f"{model_list[i]}: {patient_pred[0]}")

        return "\n".join(predicted_diagnosis)
    
    except ValueError: # handling specific data value issues such as invalid input types or incorrect values
        return "Error: Invalid input values. Please ensure that all inputs are numbers within realistic ranges."
    
    except Exception as e: # handling other unexpected errors
        return (
            f"An error occurred while processing your request. Error: {str(e)}. "
            "Please verify the input values and try again. If the problem persists, contact the administrator."
        )

def visualize_user_data_distribution(*args):
    try:
        patient_data = pd.DataFrame([args[:-1]], columns = original_columns)
        selected_diagnosis = args[-1]

        # Filtering only rows with the selected diagnosis
        selected_data = X_train_resampled_df[y_train_resampled_df['Diagnosis'] == selected_diagnosis]
        fig, axs = plt.subplots(len(original_columns), 1, figsize = (10, len(original_columns) * 3))
        fig.tight_layout(pad = 5.0)
        
        # Iterating over each feature to create the comparison plot
        for i, col in enumerate(original_columns):
            axs[i].hist(selected_data[col], bins = 30, alpha = 0.6, label = 'Healthy Distribution')
            axs[i].axvline(patient_data[col][0], color = 'r', linestyle = '--', label = 'User Input')
            axs[i].set_title(f"Feature: {col}")
            axs[i].legend()
            axs[i].set_xlabel(col)
            axs[i].set_ylabel("Frequency")
        
        plot_path = 'feature_distributions.png'
        plt.savefig(plot_path, bbox_inches = 'tight')
        plt.close()
        
        return plot_path
    
    except ValueError:
        return "Error: Invalid input values. Please ensure that all inputs are numbers within realistic ranges."
    
    except Exception as e:
        return (
            f"An error occurred while visualizing the data. Error: {str(e)}. "
            "Please verify the input values and try again. If the problem persists, contact the administrator."
        )

def generate_pdf_report(user_inputs, prediction):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)

        pdf.cell(200, 10, txt = "Anemia Diagnosis Prediction Report", ln = True, align = 'C')
        pdf.ln(10)
        pdf.cell(200, 10, txt = "User Inputs:", ln = True)
        for col, val in zip(original_columns, user_inputs[:-1]):
            pdf.cell(200, 10, txt = f"{col}: {val}", ln = True)
        pdf.ln(10)
        pdf.cell(200, 10, txt = "Predicted Diagnosis:", ln = True)
        pdf.multi_cell(200, 10, txt = prediction)
        
        pdf_path = "anemia_diagnosis_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        return (
            f"An error occurred while generating the PDF report. Error: {str(e)}. "
            "Please verify the input values and try again. If the problem persists, contact the administrator."
        )

def anemia_information():
    return (
        "**What is Anemia?**\n\n"
        "Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues.\n\n"
        "**Causes of Anemia**\n\n"
        "- Iron deficiency\n"
        "- Vitamin B12 deficiency\n"
        "- Chronic diseases\n"
        "- Genetic conditions\n\n"
        "**Symptoms of Anemia**\n\n"
        "- Fatigue\n"
        "- Weakness\n"
        "- Pale or yellowish skin\n"
        "- Shortness of breath\n"
        "- Dizziness\n\n"
        "**Treatment**\n\n"
        "The treatment of anemia depends on the cause, and it may include dietary changes, supplements, medications, or other therapies."
    )

# Gradio interface
iface = gr.Interface(
        fn = lambda *args: (
            predict_diagnosis(*args), 
            visualize_user_data_distribution(*args),
            generate_pdf_report(args, predict_diagnosis(*args))
        ),
        inputs = input_fields,
    outputs = [
        gr.Textbox(label = "Predictions"),  # the Gradio component for the prediction output
        gr.Image(type = "filepath", label = "Feature Distributions"), # the Gradio component for visualizing feature distributions
        gr.File(label = "Download PDF Report")
    ],
    title = "Anemia Diagnosis Prediction",
    description = (
        "Enter your blood test results to get a diagnosis prediction from various models. \n\n"
        "**Disclaimer:** \n" 
        "The predictions provided are for informational purposes only and should not be considered "
        "medical advice. Always consult a healthcare professional for any medical concerns."
    ),
    live = False
)

# Adding anemia information as a separate, always-visible component
iface_anemia_info = gr.Interface(
    fn = None,
    inputs = [],
    outputs = gr.Markdown(value = anemia_information(), label = "Anemia Information"),
    title = "Anemia Information"
)

# Launching both interfaces
gr.TabbedInterface([iface, iface_anemia_info], tab_names = ["Diagnosis Prediction", "Anemia Information"]).launch(debug = True)