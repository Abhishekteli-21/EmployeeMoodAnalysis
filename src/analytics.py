import pandas as pd
import plotly.express as px
from src.database import MoodDatabase
from src.utils import load_config, send_email

config = load_config()

class MoodAnalytics:
    def __init__(self):
        self.db = MoodDatabase()

    def get_task_recommendation(self, employee_id):
        historical = self.db.get_historical_data(employee_id)
        if not historical:
            return "General task"
        latest = historical[-1]
        stress_score = latest["stress_score"]
        if stress_score > config["emotion_detection"]["stress_threshold"]:
            return "Take a break or light task"
        elif "happy" in (latest["video_emotion"], latest["text_emotion"], latest["speech_emotion"]):
            return "Challenging task"
        return "Routine task"

    def team_mood_summary(self):
        data = self.db.get_team_data()
        df = pd.DataFrame(data)
        return df.groupby("timestamp").agg({
            "video_emotion": lambda x: x.mode()[0] if not x.empty else "neutral",
            "stress_score": "mean"
        }).reset_index()

    def check_stress_alert(self, employee_id):
        historical = self.db.get_historical_data(employee_id)
        if len(historical) < config["emotion_detection"]["alert_duration"]:
            return False
        recent = historical[-config["emotion_detection"]["alert_duration"]:]
        avg_stress = sum(row["stress_score"] for row in recent) / len(recent)
        if avg_stress > config["emotion_detection"]["stress_threshold"]:
            subject = f"Stress Alert for Employee {employee_id}"
            body = f"Employee {employee_id} has shown prolonged stress (avg score: {avg_stress:.2f}) over the last {len(recent)} readings."
            send_email(config["notifications"]["hr_email"], subject, body, config)
            return True
        return False

    def plot_mood_trend(self, employee_id):
        historical = self.db.get_historical_data(employee_id)
        if not historical:
            return None
        df = pd.DataFrame(historical)
        fig = px.line(df, x="timestamp", y="stress_score", title=f"Stress Trend for {employee_id}",
                      labels={"stress_score": "Stress Level", "timestamp": "Time"},
                      color_discrete_sequence=["#FF5733"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return fig

    def plot_team_heatmap(self):
        data = self.db.get_team_data()
        if not data:
            return None
        df = pd.DataFrame(data)
        fig = px.density_heatmap(df, x="timestamp", y="video_emotion", z="stress_score",
                                 title="Team Mood Heatmap", color_continuous_scale="RdYlGn_r")
        return fig