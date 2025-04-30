import os, re, duckdb, spacy, openai
from transformers import pipeline
from autogen import LLMConfig
from autogen.agents.experimental import DeepResearchAgent

def get_openai_key():
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise RuntimeError("OPENAI_API_KEY not set")
    return k

def make_llm_config():
    return LLMConfig(api_type="openai", model="gpt-4o", api_key=get_openai_key())

def make_deep_research_agent():
    return DeepResearchAgent(name="deep_research_indore", llm_config=make_llm_config(), max_web_steps=20)

nlp = spacy.load("en_core_web_sm")
sentiment = pipeline("sentiment-analysis")

class DeepResearchIngestionAgent:
    def run(self, ctx=None):
        prompt = ("Research top social-media & news issues in Indore district "
                  "around governance, civic services, public grievances.")
        return make_deep_research_agent().run({"query": prompt})

class PreProcessingAgent:
    def run(self, recs):
        for r in recs:
            r["cleaned"] = re.sub(r'[^A-Za-z0-9\\s]', '', r["text"]).strip()
        return recs

class SentimentTrendAgent:
    def run(self, recs):
        for r in recs:
            r["sentiment"] = sentiment(r["cleaned"])[0]["label"]
        return recs

class IssueIdentificationAgent:
    def run(self, recs):
        return [r for r in recs if r["sentiment"]=="NEGATIVE"]

class StrategyBuilderAgent:
    def run(self, issues):
        return [{"topic":i["cleaned"],
                 "talking_point":f"Highlight government's slow response on '{i['cleaned']}' in Indore."}
                for i in issues]

class SpeechwritingAgent:
    def run(self, strategies):
        out=[]
        for s in strategies:
            p=f"Write 2-3 sentence Hinglish opposition snippet for: {s['topic']}. Call-to-action."
            resp = openai.ChatCompletion.create(model="gpt-4o",
                messages=[{"role":"user","content":p}], api_key=get_openai_key())
            out.append(resp.choices[0].message.content)
        return out

class CrossPlatformAgent:
    def run(self, strategies):
        posts={}
        for s in strategies:
            p=(f"Issue: {s['topic']}\\nPoint: {s['talking_point']}\\n"
               "Generate: 1) Tweet, 2) FB post, 3) Insta caption+hashtags, 4) 2-min news script.")
            resp = openai.ChatCompletion.create(model="gpt-4o",
                messages=[{"role":"user","content":p}], api_key=get_openai_key(), temperature=0.7)
            posts[s["topic"]]=resp.choices[0].message.content
        return posts

def load_social_db(path="indore.db"):
    con=duckdb.connect(path, read_only=True)
    df=con.execute("SELECT text,source,ts FROM social_media").df()
    con.close()
    return df.to_dict(orient="records")
