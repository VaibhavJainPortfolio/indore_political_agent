import streamlit as st, os
from agents import load_social_db

st.sidebar.header("Config")
k=st.sidebar.text_input("OpenAI API Key",type="password")
if k: os.environ["OPENAI_API_KEY"]=k
else: st.sidebar.warning("Enter API key"); st.stop()

from team import master
st.title("Indore Political-AI Dashboard")

if st.button("Analyze Now"):
    res=master.run()
    st.subheader("1. Ingestion")
    for i,t in enumerate(res["ingestion"] or [],1): st.write(f"{i}. {t}")
    st.subheader("2. Records"); st.dataframe(res["records"])
    st.subheader("3. Issues"); [st.write(i["cleaned"]) for i in res["issues"]]
    st.subheader("4. Strategies"); [st.write(s["talking_point"]) for s in res["strategies"]]
    st.subheader("5. Speeches"); [st.write(sp) for sp in res["speeches"]]
    st.subheader("6. Cross-Platform Posts")
    for t,p in (res["cross_posts"] or {}).items(): st.markdown(f"**{t}**"); st.text(p)
