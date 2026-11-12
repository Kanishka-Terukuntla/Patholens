import hashlib
import streamlit as st
from PIL import Image
from src.model_loader import load_model, CONFIG
from src.preprocessing import prepare_image
from src.prediction import predict
from src.gradcam import generate_gradcam
from src.bounding_boxes import activation_boxes, draw_boxes
from src.visualization import probability_figure

st.title("Interactive Image Prediction")
st.write("Upload a JPG, JPEG or PNG histopathology tile. Previews are kept compact; inference and box coordinates use the full-resolution image.")
upload=st.file_uploader("Choose an image",type=["jpg","jpeg","png"])
@st.cache_resource(show_spinner=False)
def cached_model(): return load_model()

def preview(image, max_side=420):
    """Resize a display-only copy while preserving its aspect ratio."""
    image=image.convert("RGB").copy()
    image.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    return image

if upload:
    try:
        image_bytes=upload.getvalue()
        image_key=hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("prediction_image_key") != image_key:
            source=Image.open(upload)
            original,tensor=prepare_image(source)
            if min(original.size)>=32:
                with st.spinner("Loading model and preparing class explanation…"):
                    model,device=cached_model()
                    result=predict(model,tensor,device)
                    cam=generate_gradcam(model,tensor,device,result.class_index)
                st.session_state["prediction_image_key"]=image_key
                st.session_state["prediction_analysis"]={
                    "original":original,
                    "class_index":result.class_index,
                    "confidence":result.confidence,
                    "probabilities":result.probabilities.cpu().numpy(),
                    "cam":cam,
                }

        analysis=st.session_state.get("prediction_analysis")
        if analysis is None or st.session_state.get("prediction_image_key") != image_key:
            original=Image.open(upload).convert("RGB")
            st.error("Image is too small for a useful demonstration (minimum side: 32 pixels).")
        else:
            original=analysis["original"]
            if min(original.size)<32:
                st.error("Image is too small for a useful demonstration (minimum side: 32 pixels).")
            else:
                names=CONFIG["class_names"]
                class_index=analysis["class_index"]
                st.success(f"Predicted class: **{names[class_index]}**")
                st.metric("Model confidence",f"{analysis['confidence']:.1%}")
                st.caption("Confidence and probabilities are uncalibrated softmax outputs from model logits. No E2-specific calibration result was saved. No clinical interpretation is implied.")
                threshold_percent=st.slider("Grad-CAM activation threshold",min_value=0,max_value=100,value=60,step=1,format="%d%%")
                st.caption("This filters approximate regions by normalized Grad-CAM activation. The classifier does not produce a separate confidence score for each region.")
                boxes=activation_boxes(analysis["cam"],original.size,threshold=threshold_percent/100)
                heat=Image.fromarray((analysis["cam"]*255).clip(0,255).astype("uint8"),mode="L").resize(original.size,Image.Resampling.BILINEAR)
                heat_color=Image.merge("RGB",(heat,Image.new("L",original.size,0),Image.new("L",original.size,0)))
                annotated=draw_boxes(original,boxes) if boxes else original.copy()
                st.subheader("Visual outputs")
                out_a,out_b,out_c=st.columns(3,gap="small")
                with out_a:
                    st.markdown("**A · Original**")
                    st.image(preview(original),use_container_width=True)
                with out_b:
                    st.markdown("**B · Grad-CAM**")
                    st.image(preview(heat_color),use_container_width=True,caption="Red indicates relative activation.")
                with out_c:
                    st.markdown("**C · Original + boxes**")
                    caption=f"{len(boxes)} region(s)" if boxes else "No meaningful region found; no box drawn."
                    st.image(preview(annotated),use_container_width=True,caption=caption)
                st.caption("Previews are capped at 300×300 pixels for display. Full-resolution coordinates are used for box mapping. Grad-CAM boxes are approximate activation regions, not detections, lesion boundaries, or segmentation masks.")
                st.subheader("Nine-class probabilities")
                st.pyplot(probability_figure(names,analysis["probabilities"]),use_container_width=True)
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else: st.info("Upload an image to begin. The model checkpoint and preprocessing configuration are packaged in this application folder.")
