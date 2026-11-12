import streamlit as st
from src.evaluation import experiment_results, validation_results, training_results
from src.visualization import metric_chart, validation_test_figure, training_curves_figure
st.title("Methodology and Experiments")
st.write("All four experiments are in the original pipeline notebook. They share 128×128 inputs, a MobileNetV3-Small backbone family, AdamW training and ImageNet normalization. E2 adds H&E optical-density perturbation during training; E3 uses Macenko normalization; E4 adds pooled multi-scale feature taps. This application packages E2 from the 50% run.")
data=experiment_results()
st.dataframe(data[["experiment_id","experiment_name","best_epoch","test_accuracy","test_macro_precision","test_macro_recall","test_macro_f1","trainable_parameters"]].rename(columns={"experiment_id":"ID","experiment_name":"Experiment","best_epoch":"Best epoch","test_accuracy":"Test accuracy","test_macro_precision":"Macro precision","test_macro_recall":"Macro recall","test_macro_f1":"Macro F1","trainable_parameters":"Parameters"}).style.format({"Test accuracy":"{:.3f}","Macro precision":"{:.3f}","Macro recall":"{:.3f}","Macro F1":"{:.3f}"}), hide_index=True, use_container_width=True)
accuracy_col,f1_col=st.columns(2,gap="large")
with accuracy_col:
    st.pyplot(metric_chart(data,"test_accuracy","Pilot test accuracy"), use_container_width=True)
with f1_col:
    st.pyplot(metric_chart(data,"test_macro_f1","Pilot test macro-F1"), use_container_width=True)
st.pyplot(validation_test_figure(validation_results(),data),use_container_width=True)
st.pyplot(training_curves_figure(training_results(),"E2"),use_container_width=True)
with st.expander("Experiment details"):
    st.markdown("**E1 · RGB baseline.** Original RGB tiles; random 90° rotations and horizontal/vertical flips; compact classifier head. Best epoch 15. Test accuracy 0.9465, macro-F1 0.9324. Highest validation macro-F1: 0.9946.\n\n**E2 · HED stain augmentation.** Same compact head with stochastic H&E optical-density perturbation during training. Best epoch 15. Test accuracy 0.9643 and macro-F1 0.9522, strongest descriptive test scores. Validation macro-F1 0.9913.\n\n**E3 · Macenko stain normalization.** Macenko reference estimated from training-subset images, then used to normalize. Best epoch 15. Test accuracy 0.8521, macro-F1 0.8209.\n\n**E4 · Multi-scale MobileNetV3-Small.** RGB input with pooled features from backbone stages 3, 6 and 12 concatenated into a compact head. Best epoch 14. Test accuracy 0.9496, macro-F1 0.9334; validation macro-F1 0.9909.")
st.subheader("Selection and interpretation")
st.write("E2 is packaged because it had the strongest saved 50% test results: macro-F1 0.952221, accuracy 0.964345, macro precision 0.951283, and macro recall 0.953557. E1 had the highest validation-selection macro-F1 (0.994599 versus E2's 0.991265). Since E2 was chosen after comparing test scores, the test set has been used for model selection; its metrics here are post-selection and not an unbiased final estimate. E2's calibration metrics were not saved, so the app displays raw softmax confidence only.")
st.caption("Test metrics describe 3,590 samples from the 50% pilot subset. E2 has 1,039,721 trainable parameters and about 4.09 MB saved weights. Inference speed depends on hardware.")
