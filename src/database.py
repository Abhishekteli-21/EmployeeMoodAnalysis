from pymongo import MongoClient
from datetime import datetime
from src.utils import anonymize_employee_id, load_config

config = load_config()

class MoodDatabase:
    def __init__(self):
        self.client = MongoClient(config["database"]["uri"])
        self.db = self.client[config["database"]["db_name"]]
        self.collection = self.db[config["database"]["collection"]]

    def insert_mood(self, employee_id, video_emotion, text_emotion, speech_emotion, stress_score):
        anon_id = anonymize_employee_id(employee_id, config["privacy"]["anonymization_key"], config["privacy"]["salt_timestamp"])
        mood_entry = {
            "employee_id": anon_id,
            "timestamp": datetime.now().isoformat(),
            "video_emotion": video_emotion,
            "text_emotion": text_emotion,
            "speech_emotion": speech_emotion,
            "stress_score": stress_score
        }
        self.collection.insert_one(mood_entry)

    def get_historical_data(self, employee_id):
        anon_id = anonymize_employee_id(employee_id, config["privacy"]["anonymization_key"], False)
        return list(self.collection.find({"employee_id": anon_id}).sort("timestamp", 1))

    def get_team_data(self):
        return list(self.collection.find().sort("timestamp", 1))

    def close(self):
        self.client.close()