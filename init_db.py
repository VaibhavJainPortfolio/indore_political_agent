import duckdb, os

DB = os.path.join(os.path.dirname(__file__), "indore.db")
con = duckdb.connect(DB)
con.execute("CREATE SEQUENCE IF NOT EXISTS seq_social_media START 1")
con.execute(\"\"\"\nCREATE TABLE IF NOT EXISTS social_media (\n  id INTEGER PRIMARY KEY DEFAULT nextval('seq_social_media'),\n  text VARCHAR,\n  source VARCHAR,\n  ts TIMESTAMP\n)\n\"\"\")
cnt = con.execute("SELECT count(*) FROM social_media").fetchone()[0]
if cnt == 0:
    con.execute("INSERT INTO social_media (text, source, ts) VALUES "
                "('Waterlogging issue in Indore after rain','twitter',CURRENT_TIMESTAMP),"
                "('Job fair announced at Holkar Stadium','news',CURRENT_TIMESTAMP)")
con.close()
print("✅ Database initialized at", DB)
