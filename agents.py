# agents.py

import os
import re
import duckdb
import spacy
import openai
from transformers import pipeline

# Helper to get OpenAI key
def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key

# Load spaCy model once
_spacy = None
def get_spacy():
    global _spacy
    if not _spacy:
        try:
            import en_core_web_sm
            _spacy = en_core_web_sm.load()
        except ImportError:
            spacy.cli.download("en_core_web_sm")
            _spacy = spacy.load("en_core_web_sm")
    return _spacy

nlp = get_spacy()
sentiment_pipe = pipeline("sentiment-analysis")

# We’ll ignore DeepResearch; always fallback to DB
class DeepResearchIngestionAgent:
    def run(self, ctx=None):
        from agents import load_social_db
        return [r["text"] for r in load_social_db()]

class PreProcessingAgent:
    def run(self, recs):
        for r in recs:
            r["cleaned"] = re.sub(r"[^A-Za-z0-9\\s]", "", r["text"])
        return recs

class SentimentTrendAgent:
    def run(self, recs):
        for r in recs:
            r["sentiment"] = sentiment_pipe(r["cleaned"])[0]["label"]
        return recs

class IssueIdentificationAgent:
    def run(self, recs):
        return [r for r in recs if r["sentiment"] == "NEGATIVE"]

class StrategyBuilderAgent:
    def run(self, issues):
        return [{"topic":i["cleaned"],
                 "talking_point":f"Highlight govt’s slow response on '{i['cleaned']}'."}
                for i in issues]

class SpeechwritingAgent:
    def run(self, strategies):
        out=[]
        for s in strategies:
            p=f"Write 2-3 sentence snippet for: {s['topic']}."
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role":"user","content":p}],
                api_key=get_openai_key()
            )
            out.append(resp.choices[0].message.content)
        return out

class CrossPlatformAgent:
    def run(self, strategies):
        return {}   # skip on Cloud

def load_social_db(db_path="indore.db"):
    con=duckdb.connect(db_path, read_only=True)
    df=con.execute("SELECT text,source,ts FROM social_media").df()
    con.close()
    return df.to_dict(orient="records")
