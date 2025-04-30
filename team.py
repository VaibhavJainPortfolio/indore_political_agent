from datetime import datetime
from agents import (DeepResearchIngestionAgent, PreProcessingAgent,
                    SentimentTrendAgent, IssueIdentificationAgent,
                    StrategyBuilderAgent, SpeechwritingAgent,
                    CrossPlatformAgent)

class MasterAgent:
    def __init__(self):
        self.ingest=DeepResearchIngestionAgent()
        self.prep=PreProcessingAgent()
        self.sent=SentimentTrendAgent()
        self.issues=IssueIdentificationAgent()
        self.strat=StrategyBuilderAgent()
        self.speech=SpeechwritingAgent()
        self.cross=CrossPlatformAgent()

    def run(self):
        ing=self.ingest.run()
        raw=getattr(ing,"report",None) or getattr(ing,"output",None) or []
        recs=[{"text":t,"source":"deep_research","ts":datetime.now()} for t in raw]
        recs=self.prep.run(recs)
        recs=self.sent.run(recs)
        iss=self.issues.run(recs)
        strat=self.strat.run(iss)
        sp=self.speech.run(strat)
        cp=self.cross.run(strat)
        return {"ingestion":raw, "records":recs, "issues":iss,
                "strategies":strat, "speeches":sp, "cross_posts":cp}

master=MasterAgent()
