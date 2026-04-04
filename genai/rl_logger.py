# import os
# import sqlite3
# import json
# from datetime import datetime

# class RLExperienceLogger:
#     def __init__(self):
#         # GET THE ABSOLUTE PATH TO PROJECT ROOT
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(current_dir)
#         self.db_path = os.path.join(project_root, "rl_feedback.db")
#         self._init_db()

#     def _init_db(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS experiences (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     query TEXT,
#                     query_vector TEXT,
#                     selected_chunk_ids TEXT,
#                     reward REAL,
#                     timestamp TEXT
#                 )
#             ''')
#             conn.commit()

#     def log_experience(self, query, q_vec, ids, reward):
#         # Force the path to be the absolute root
#         db_path = os.path.join(os.getcwd(), "rl_feedback.db")
#         with sqlite3.connect(db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute('''
#                 INSERT INTO experiences (query, query_vector, selected_chunk_ids, reward, timestamp)
#                 VALUES (?, ?, ?, ?, ?)
#             ''', (query, json.dumps(q_vec), json.dumps(ids), reward, datetime.now().isoformat()))
#             conn.commit()
#         # This MUST appear in your terminal when you click the button
#         print(f"💰 DATA SAVED: Reward {reward} recorded in {db_path}")

#     def get_stats(self):
#         db_path = os.path.join(os.getcwd(), "rl_feedback.db")
#         if not os.path.exists(db_path):
#             return 0, 0.0
#         with sqlite3.connect(db_path) as conn:
#             res = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()
#             count = res[0] if res else 0
#             res_avg = conn.execute("SELECT AVG(reward) FROM experiences").fetchone()
#             avg = res_avg[0] if res_avg[0] is not None else 0.0
#             return count, avg

# if __name__ == "__main__":
#     logger = RLExperienceLogger()
#     print("RL Logger initialized.")
#     count, avg = logger.get_stats()
#     print(f"Current Training Data: {count} samples, Avg Reward: {avg}")

import os
import sqlite3
import json
from datetime import datetime


class RLExperienceLogger:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.db_path = os.path.join(project_root, "rl_feedback.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT,
                    query TEXT,
                    query_vector TEXT,
                    selected_chunk_ids TEXT,
                    reward REAL,
                    timestamp TEXT
                )
            ''')

            # Migration support: add project_name to old DBs if missing
            cols = [row[1] for row in cursor.execute("PRAGMA table_info(experiences)").fetchall()]
            if "project_name" not in cols:
                cursor.execute("ALTER TABLE experiences ADD COLUMN project_name TEXT")

            conn.commit()

    def log_experience(self, project_name, query, q_vec, ids, reward):
        db_path = os.path.join(os.getcwd(), "rl_feedback.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiences (
                    project_name, query, query_vector, selected_chunk_ids, reward, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                project_name,
                query,
                json.dumps(q_vec),
                json.dumps(ids),
                reward,
                datetime.now().isoformat()
            ))
            conn.commit()

        print(f"💰 DATA SAVED: project={project_name}, reward={reward} recorded in {db_path}")

    def get_stats(self, project_name=None):
        db_path = os.path.join(os.getcwd(), "rl_feedback.db")
        if not os.path.exists(db_path):
            return 0, 0.0

        with sqlite3.connect(db_path) as conn:
            if project_name:
                res = conn.execute(
                    "SELECT COUNT(*), AVG(reward) FROM experiences WHERE project_name=?",
                    (project_name,)
                ).fetchone()
            else:
                res = conn.execute(
                    "SELECT COUNT(*), AVG(reward) FROM experiences"
                ).fetchone()

            count = res[0] if res else 0
            avg = res[1] if res and res[1] is not None else 0.0
            return count, avg


if __name__ == "__main__":
    logger = RLExperienceLogger()
    print("RL Logger initialized.")
    count, avg = logger.get_stats()
    print(f"Current Training Data: {count} samples, Avg Reward: {avg}")