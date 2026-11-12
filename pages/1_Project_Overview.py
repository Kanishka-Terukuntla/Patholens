import streamlit as st
from src.evaluation import training_class_counts
from src.visualization import class_distribution_figure
st.title("Project Overview")
st.write("PathoLens AI classifies 128×128 colorectal histopathology tiles from PathMNIST into nine tissue categories. It packages E2, the top performer on the saved test comparison, and includes class-score visualization and approximate Grad-CAM activation localization.")
st.warning("This pilot is not clinically validated. It does not diagnose cancer or provide validated lesion localization.")
st.subheader("Problem and objectives")
st.write("Histology review involves distinguishing tissue patterns. This project explores a compact computer-vision workflow, compares four training recipes, and presents model outputs for inspection rather than replacing expert review.")
st.subheader("Nine PathMNIST categories")
st.write("Adipose · background · debris · lymphocytes · mucus · smooth muscle · normal colon mucosa · cancer-associated stroma · colorectal adenocarcinoma epithelium")
st.subheader("System workflow")
st.code("Uploaded RGB image → 128×128 resize and ImageNet normalization → E2 MobileNetV3-Small with HED stain augmentation training → nine class probabilities → Grad-CAM → connected high-activation regions → boxes on a fresh copy of the original")
st.subheader("Dataset and scope")
st.write("Saved run metadata describes a stratified 50% pilot subset of each official PathMNIST split (seed 42): 44,998 train, 5,002 validation, and 3,590 test images. Validation was divided into 3,501 model-selection and 1,501 calibration examples. Results are pilot-subset results, not full benchmark results.")
st.subheader("Training split class distribution")
class_names=["Adipose","Background","Debris","Lymphocytes","Mucus","Smooth muscle","Normal mucosa","Cancer stroma","Adenocarcinoma"]
left, chart, right = st.columns([0.08, 0.84, 0.08])
with chart:
    st.pyplot(class_distribution_figure(training_class_counts(),class_names),use_container_width=True)
