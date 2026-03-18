import sqlite3

conn = sqlite3.connect('freelancer_dss.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Jobs_Core (
        job_id INTEGER PRIMARY KEY,
        title TEXT,
        category TEXT,
        subcategory TEXT,
        rating REAL,
        votes REAL,
        is_capped INTEGER
    )
''')

cursor.execute('''

    CREATE TABLE IF NOT EXISTS HNN_Predictions (
        job_id INTEGER,
        mu_price REAL,
        sigma_uncertainty REAL,
        tier_recommendation TEXT,
        top_features TEXT,
        FOREIGN KEY (job_id) REFERENCES Jobs_Core (job_id)
    )

''')

