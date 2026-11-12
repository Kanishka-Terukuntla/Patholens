import streamlit as st

st.markdown('<div class="hero"><h1>PathoLens AI</h1><p>Colorectal tissue classification with transparent, research-focused visual explanations</p></div>', unsafe_allow_html=True)
st.write("Classify a colorectal tissue tile, explore model evaluation results, or inspect class-associated image regions.")
st.info("For research and education. Predictions are not medical diagnoses, and Grad-CAM regions are not validated lesion localizations.")
st.markdown("### Project workflow")
cols = st.columns(5)
for col, title, detail in zip(cols, ["Image", "Preprocess", "Classify", "Explain", "Review"], ["Upload an image", "Resize to 128² · RGB · ImageNet normalization", "Nine tissue-class scores", "Grad-CAM + approximate regions", "Interpret with limitations"]):
    with col:
        st.markdown(f"**{title}**")
        st.caption(detail)
st.markdown("### Current application model")
left, right = st.columns(2)
left.metric("Best observed test model", "E2 · HED stain augmentation", "test macro-F1 0.9522")
right.metric("E2 test accuracy", "0.9643", "3,590 test images")
st.caption("E2 is selected based on the saved test comparison. Those test metrics are post-selection and should not be treated as an unbiased final estimate. See Methodology for details.")
