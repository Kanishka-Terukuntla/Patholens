"""PathoLens AI entry point with optional standalone project guide route."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
FLOWCHART = ROOT / "docs" / "PathoLens_Project_Flowchart.html"

st.set_page_config(page_title="PathoLens AI", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"], .stApp {background:#080D19;color:#E8EDF7}
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {background:#080D19;color:#E8EDF7}
[data-testid="stMainBlockContainer"] {max-width:1500px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stHeader"] {background:rgba(8,13,25,.94);color:#E8EDF7}
[data-testid="stToolbar"] button {color:#D8E1F0}
[data-testid="stAppDeployButton"], #MainMenu {display:none !important}
[data-testid="stSidebar"] {background:#0D1526;border-right:1px solid #202B43}
[data-testid="stSidebar"] * {color:#E8EDF7}
.hero {padding:1.8rem 2rem;border-radius:20px;background:linear-gradient(118deg,#14213A 0%,#153C50 55%,#14545A 100%);color:#F6F8FC;margin-bottom:1.5rem;border:1px solid #2B5268;box-shadow:0 18px 45px rgba(0,0,0,.18)}
.hero h1 {margin:0;color:#F6F8FC;font-size:2.45rem;letter-spacing:-.035em}.hero p {margin:.5rem 0 0;color:#C7D6E8;font-size:1.05rem}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#121D32,#111A2B);border:1px solid #26334C;padding:1rem 1.15rem;border-radius:15px;color:#E8EDF7}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {color:#A9B8D0 !important;opacity:1 !important}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {color:#F3F6FC !important;opacity:1 !important}
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {color:#66D8C5 !important;opacity:1 !important}
[data-testid="stAlert"] {background:#111D32;border:1px solid #2A3B59;color:#E8EDF7;border-radius:12px}
[data-testid="stFileUploader"] {background:#111A2D;border:1px dashed #455A7B;border-radius:14px;padding:.5rem}
[data-testid="stExpander"] {background:#10192A;border:1px solid #26334C;border-radius:12px}
[data-testid="stDataFrame"], [data-testid="stTable"] {background:#10192A}
[data-testid="stDataFrame"] [role="columnheader"] {background:#192640;color:#DCE5F3}
[data-testid="stSidebarNav"] a[aria-current="page"] {background:#1A2C47;border-radius:9px}
[data-testid="stSidebarNav"] a:hover {background:#17243B;border-radius:9px}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {border-radius:8px}
div[data-testid="stMarkdownContainer"] h1,div[data-testid="stMarkdownContainer"] h2,div[data-testid="stMarkdownContainer"] h3 {letter-spacing:-.02em}
::selection {background:#5968A8;color:#FFFFFF}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 PathoLens AI")
    st.caption("Histopathology · Image analysis")
    st.markdown("---")
    st.caption("9 tissue classes · 128×128")

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Project_Overview.py", title="Project Overview", icon="🔬"),
    st.Page("pages/2_Methodology_and_Experiments.py", title="Methodology and Experiments", icon="📊"),
    st.Page("pages/3_Test_Results_and_Best_Model.py", title="Test Results and Best Model", icon="📈"),
    st.Page("pages/4_Image_Prediction.py", title="Image Prediction", icon="🧪"),
    st.Page("pages/5_Explainability.py", title="Explainability", icon="🧭"),
]

if FLOWCHART.is_file():
    pages.append(st.Page("pages/6_Project_Flowchart.py", title="Project Flowchart", icon="🗺️"))

navigation = st.navigation(pages, position="sidebar")
navigation.run()
