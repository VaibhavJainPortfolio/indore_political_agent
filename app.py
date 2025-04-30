# app.py
import streamlit as st
import os
from agents import load_social_db
from team import master

# 1️⃣ Sidebar: OpenAI API key input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password",
    help="Paste your sk-… key here to enable ChatGPT-powered agents"
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Enter your OpenAI API key to proceed.")
    st.stop()

st.title("🗳️ Indore Political-AI Swarm Dashboard")

# 2️⃣ Cache the full analysis so it's only run once per session/key
@st.cache_data(show_spinner=False)
def run_analysis():
    return master.run()

# 3️⃣ Cache fallback DB reads
@st.cache_data(show_spinner=False)
def get_fallback_records():
    return load_social_db()

if st.button("Analyze Now"):
    # Run the agent swarm
    result = run_analysis()

    # 1. Ingestion
    st.subheader("1. Deep-Research Ingestion Agent")
    raw = result.get("ingestion") or []
    if raw:
        for i, txt in enumerate(raw, 1):
            st.write(f"{i}. {txt}")
    else:
        st.info("No Deep-Research insights; loading fallback DB data.")
        raw = [r["text"] for r in get_fallback_records()]
        for i, txt in enumerate(raw, 1):
            st.write(f"{i}. {txt}")

    # 2. Records with sentiment
    st.subheader("2. Preprocessing & Sentiment Agent")
    records = result.get("records") or get_fallback_records()
    st.dataframe(records)

    # 3. Identified Issues
    st.subheader("3. Issue Identification Agent")
    issues = result.get("issues") or []
    if issues:
        for issue in issues:
            st.write("•", issue.get("cleaned"))
    else:
        st.write("No negative-sentiment issues flagged.")

    # 4. Strategy Builder
    st.subheader("4. Strategy Builder Agent")
    strategies = result.get("strategies") or []
    if strategies:
        for s in strategies:
            st.write("•", s.get("talking_point"))
    else:
        st.write("No strategies generated.")

    # 5. Speechwriting
    st.subheader("5. Speechwriting Agent")
    speeches = result.get("speeches") or []
    if speeches:
        for idx, sp in enumerate(speeches, 1):
            st.write(f"Snippet {idx}: {sp}")
    else:
        st.write("No speech snippets generated.")

    # 6. Cross-Platform Posts
    st.subheader("6. CrossPlatform Agent Outputs")
    cross = result.get("cross_posts") or {}
    if cross:
        for topic, content in cross.items():
            st.markdown(f"**{topic}**")
            st.text(content)
    else:
        st.write("No cross-platform posts generated.")
