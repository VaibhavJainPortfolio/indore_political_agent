import os
import re
import duckdb
import spacy
import openai
from transformers import pipeline

# Try importing DeepResearchAgent; if unavailable, we’ll fallback
try:
    from autogen import LLMConfig
    from autogen.agents.experimental import DeepResearchAgent
    HAVE_DEEP = True
except ImportError:
    HAVE_DEEP = False

# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_openai_key():
    # Read from environment (Streamlit secrets automatically injected into env)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Define it in Streamlit secrets or env.")
    return key

# ─── Factories ─────────────────────────────────────────────────────────────────

def make_deep_research_agent():
    if not HAVE_DEEP:
        raise RuntimeError("DeepResearchAgent unavailable in this environment.")
    llm_cfg = LLMConfig(api_type="openai", model="gpt-4o", api_key=get_openai_key())
    return DeepResearchAgent(name="deep_research_indore", llm_config=llm_cfg, max_web_steps=20)

# ─── NLP & Sentiment Setup ──────────────────────────────────────────────────────

# Load small English model; spaCy is in requirements
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentiment_pipe = pipeline("sentiment-analysis")

# ─── Agent Classes ─────────────────────────────────────────────────────────────

class DeepResearchIngestionAgent:
    """Fetches issues via DeepResearchAgent or falls back to DuckDB data."""
    def run(self, context=None):
        if HAVE_DEEP:
            prompt = (
                "Research the top social-media and news issues in Indore district "
                "around governance, civic services, and public grievances."
            )
            agent = make_deep_research_agent()
            resp = agent.run({"query": prompt})
            # resp may have .report or .output
            if hasattr(resp, 'report'):
                return list(resp.report)
            if hasattr(resp, 'output'):
                return list(resp.output)
        # fallback to local DB
        from agents import load_social_db
        return [r['text'] for r in load_social_db()]

class PreProcessingAgent:
    """Clean and normalize text entries."""
    def run(self, records):
        for r in records:
            txt = r.get('text','')
            clean = re.sub(r'[^A-Za-z0-9\s]', '', txt).strip()
            r['cleaned'] = clean
        return records

class SentimentTrendAgent:
    """Apply sentiment analysis."""
    def run(self, records):
        for r in records:
            r['sentiment'] = sentiment_pipe(r['cleaned'])[0]['label']
        return records

class IssueIdentificationAgent:
    """Flag negative-sentiment entries as issues."""
    def run(self, records):
        return [r for r in records if r.get('sentiment')=='NEGATIVE']

class StrategyBuilderAgent:
    """Create opposition talking points."""
    def run(self, issues):
        return [{
            'topic': i['cleaned'],
            'talking_point': f"Highlight government's slow response on '{i['cleaned']}' in Indore."
        } for i in issues]

class SpeechwritingAgent:
    """Generate speech snippets via OpenAI."""
    def run(self, strategies):
        snippets=[]
        for s in strategies:
            prompt = f"Write 2-3 sentence Hinglish opposition snippet for: {s['topic']}. Include call-to-action."
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{'role':'user','content':prompt}],
                api_key=get_openai_key()
            )
            snippets.append(resp.choices[0].message.content)
        return snippets

class CrossPlatformAgent:
    """Produce posts for X, Facebook, Instagram and a news-script."""
    def run(self, strategies):
        posts={}
        for s in strategies:
            prompt = (
                f"Issue: {s['topic']}\nPoint: {s['talking_point']}\n"
                "Generate: 1) Tweet, 2) Facebook post (~100 words), 3) Instagram caption with 5 hashtags, 4) 2-min news script."
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{'role':'user','content':prompt}],
                api_key=get_openai_key(),
                temperature=0.7
            )
            posts[s['topic']] = resp.choices[0].message.content
        return posts

# ─── Fallback DB loader ─────────────────────────────────────────────────────────

def load_social_db(db_path="indore.db"):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT text, source, ts FROM social_media").df()
    con.close()
    return df.to_dict(orient='records')
