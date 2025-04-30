# agents.py
import os
import re
import duckdb
import spacy
import openai
from transformers import pipeline

# Attempt to import DeepResearchAgent; fallback if missing
try:
    from autogen import LLMConfig
    from autogen.agents.experimental import DeepResearchAgent
    HAVE_DEEP = True
except ImportError:
    HAVE_DEEP = False

# Helper: get OpenAI key at runtime
def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it in Streamlit sidebar or secrets.")
    return key

# Factory for DeepResearchAgent
def make_deep_research_agent():
    if not HAVE_DEEP:
        raise RuntimeError("DeepResearchAgent unavailable in this environment.")
    cfg = LLMConfig(api_type="openai", model="gpt-4o", api_key=get_openai_key())
    return DeepResearchAgent(name="deep_research_indore", llm_config=cfg, max_web_steps=5)

# Load spaCy model once
_spacy_model = None
def load_spacy():
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            _spacy_model = spacy.load("en_core_web_sm")
    return _spacy_model

nlp = load_spacy()
sentiment_pipe = pipeline("sentiment-analysis")

class DeepResearchIngestionAgent:
    def run(self, context=None):
        if HAVE_DEEP:
            prompt = (
                "Research the top social-media and news issues in Indore district "
                "around governance, civic services, and public grievances."
            )
            resp = make_deep_research_agent().run({"query": prompt})
            if hasattr(resp, 'report'):
                return list(resp.report)
            if hasattr(resp, 'output'):
                return list(resp.output)
        # Fallback: load from DuckDB
        from agents import load_social_db
        return [r['text'] for r in load_social_db()]

class PreProcessingAgent:
    def run(self, records):
        for r in records:
            r['cleaned'] = re.sub(r'[^A-Za-z0-9\s]', '', r.get('text','')).strip()
        return records

class SentimentTrendAgent:
    def run(self, records):
        for r in records:
            r['sentiment'] = sentiment_pipe(r['cleaned'])[0]['label']
        return records

class IssueIdentificationAgent:
    def run(self, records):
        return [r for r in records if r.get('sentiment')=='NEGATIVE']

class StrategyBuilderAgent:
    def run(self, issues):
        return [{
            'topic':i['cleaned'],
            'talking_point':f"Highlight government's slow response on '{i['cleaned']}' in Indore."
        } for i in issues]

class SpeechwritingAgent:
    def run(self, strategies):
        outs=[]
        for s in strategies:
            p=f"Write a 2-3 sentence Hinglish opposition snippet for: {s['topic']}. Call-to-action."
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{'role':'user','content':p}],
                api_key=get_openai_key()
            )
            outs.append(resp.choices[0].message.content)
        return outs

class CrossPlatformAgent:
    def run(self, strategies):
        posts={}
        for s in strategies:
            p=(f"Issue: {s['topic']}\nPoint: {s['talking_point']}\n"
               "Generate: 1) Tweet, 2) Facebook post, 3) Instagram caption+hashtags, 4) 2-min news script.")
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{'role':'user','content':p}],
                api_key=get_openai_key(), temperature=0.7
            )
            posts[s['topic']] = resp.choices[0].message.content
        return posts

# Fallback loader
def load_social_db(db_path="indore.db"):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT text, source, ts FROM social_media").df()
    con.close()
    return df.to_dict(orient='records')
