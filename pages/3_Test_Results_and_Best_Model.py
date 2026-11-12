from pathlib import Path
import streamlit as st
from src.evaluation import experiment_results, per_class_results, confusion_results
from src.visualization import confusion_figure
root=Path(__file__).resolve().parents[1]
st.title("Test Results and Best Model")
data=experiment_results(); row=data[data.experiment_id=="E2"].iloc[0]
cols=st.columns(4)
for col,label,key in zip(cols,["Accuracy","Macro precision","Macro recall","Macro F1"],["test_accuracy","test_macro_precision","test_macro_recall","test_macro_f1"]): col.metric(label,f"{row[key]:.4f}")
st.caption("E2 is selected as best observed test performer. 50% pilot test n=3,590; because these results informed selection, they are post-selection.")
st.subheader("Per-class report · E2")
pc=per_class_results("E2")
st.dataframe(pc[["class_name","support","precision","recall","f1"]].rename(columns={"class_name":"Class","support":"Support","precision":"Precision","recall":"Recall","f1":"F1"}).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}),hide_index=True,use_container_width=True)
st.subheader("Confusion matrix")
left, chart, right = st.columns([0.08, 0.84, 0.08])
with chart:
    st.pyplot(confusion_figure(confusion_results("E2")),use_container_width=True)
st.subheader("Calibration availability")
st.info("No E2-specific calibration or selective-prediction results were saved. Prediction probabilities and confidence are raw softmax outputs. E1 calibration values are not applied to E2.")
st.subheader("Why E2 is packaged")
st.write("E2 leads the saved test comparison in accuracy (0.964345), macro precision (0.951283), macro recall (0.953557), and macro-F1 (0.952221), with roughly the same model size as E1. E1 had the top validation-selection macro-F1. Selecting E2 after reviewing test results makes the test comparison post-selection; a new untouched evaluation set is needed for an unbiased estimate.")
st.markdown("**Pilot limitations:** a 50% subset, not the complete benchmark; patient- or slide-level independence is not established; class-wise errors remain; no external validation.")
