from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

flowchart = Path(__file__).resolve().parents[1] / "docs" / "PathoLens_Project_Flowchart.html"
st.title("PathoLens AI · Project Flowchart")
if not flowchart.is_file():
    st.info("The optional flowchart file is not present. The rest of the app works without it.")
    st.stop()

html = flowchart.read_text(encoding="utf-8")
components.html(html, height=2200, scrolling=True)
