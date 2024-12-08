import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import re

import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, pipeline

class JournalInsightGenerator:
    def __init__(self): 
        load_dotenv()
        GEMINI_KEY = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=GEMINI_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.chat = None
        self.history = []

    def one_shot_prompt(self):
        initial_prompt = """
        Imagine you are analyzing a series of journal entries for mental health insights. I will provide you with several text entries, and you need to analyze them according to the following metrics. Provide a score or summary for each metric along with a brief justification:
        
        1. *Sentiment Analysis*:
            - Rate the overall sentiment of each entry as Positive, Neutral, or Negative.
            - Provide a sentiment breakdown (e.g., Positive: 30%, Neutral: 50%, Negative: 20%).
            - Highlight any strong emotional tones (e.g., anger, sadness, joy).

        2. *Topic Analysis*:
            - Identify dominant topics in each entry (e.g., work, relationships, health).
            - List the top three topics and their relevance percentage.

        3. *Mood Correlation*:
            - Correlate sentiments and topics with previous entries to detect patterns or recurring themes.
            - Highlight any common triggers for mood changes.

        4. *Metrics for Visualization*:
            - Generate data points for visual representation, such as:
                - Sentiment trend over time.
                - Topic frequency distribution.
                - Correlation matrix between mood and topic.

        At the end, provide an overall summary of the user’s mental health trends, highlighting any significant shifts, recurrent patterns, or key insights that may be useful for further exploration.
        """

        one_shot_entry = "I felt stressed today because of the workload. I had multiple deadlines and little time to relax."
        one_shot_output = """
        Sentiment Analysis:
            - Overall Sentiment: Negative
            - Breakdown: Negative: 80%, Neutral: 20%
            - Emotional Tone: Stress, Anxiety

        Topic Analysis:
            - Dominant Topics: Work (60%), Deadlines (30%), Relaxation (10%)

        Mood Correlation:
            - Recurrent Theme: High stress associated with work and deadlines
            - Trigger: Work-related deadlines and lack of time management

        Visualization Metrics:
            - Sentiment Score: -0.6 (Negative)
            - Topic Frequency: Work (60%), Deadlines (30%), Relaxation (10%)
            - Mood Correlation Coefficient with "Work": 0.8
        
        Overall Summary:
            - The user shows high levels of stress associated with workload and deadlines. The recurring pattern suggests that time management could help alleviate stress.
        """
        # Start the chat with the one-shot example to provide context
        self.chat = self.model.start_chat(
            history=[
                {"role": "user", "parts": [initial_prompt, one_shot_entry]},
                {"role": "model", "parts": one_shot_output}
            ]
        )
        print("Initialized chat with one-shot prompt")

    def add_entry_to_history(self, entry, insights):
        """Adds a journal entry and its insights to the history for future reference."""
        self.history.append({
            "entry": entry,
            "insights": insights
        })

    def generate_history_summary(self):
        """Generates a brief summary of historical insights to use as context in the prompt."""
        summary = []
        for item in self.history[-5:]:  # Limit to the last 5 entries to keep the prompt manageable
            summary.append(f"Entry: {item['entry']}\nInsights: {item['insights']}")
        return "\n\n".join(summary)

    def prompt_gemini_for_journal(self, entry, additional_data=None):
        # Generate a summary of the recent history for context
        history_summary = self.generate_history_summary()        
        request = f"Here is a new journal entry:\n{entry}\n\nRecent journal history:\n{history_summary}{additional_data}\n\nProvide insights following the format used before."
        # Send the request to the model with the updated history and additional insights
        response = self.chat.send_message(request)
        insights = response.text
        self.add_entry_to_history(entry, insights)
        return insights


    def parse_insights(self, lines):
        insights = {
            "sentiment_analysis": {},
            "topic_analysis": {},
            "mood_correlation": {},
            "visualization_metrics": {},
            "summary": ""
        }

        for i, line in enumerate(lines):
            line = line.strip()

            # Sentiment Analysis
            if "**1. Sentiment Analysis:**" in line:
                insights["sentiment_analysis"]["overall_sentiment"] = re.search(r"\*\*Overall Sentiment:\*\* (.+)", lines[i + 1]).group(1)
                insights["sentiment_analysis"]["breakdown"] = {}
                breakdown_match = re.findall(r"\*\*Breakdown:\*\* (.+): (\d+%)", lines[i + 2])
                for label, percentage in breakdown_match:
                    insights["sentiment_analysis"]["breakdown"][label.lower()] = percentage
                strong_emotions = re.search(r"\*\*Strong Emotional Tones:\*\* (.+)", lines[i + 3])
                if strong_emotions:
                    insights["sentiment_analysis"]["strong_emotional_tones"] = [emotion.strip() for emotion in strong_emotions.group(1).split(",")]

            # Topic Analysis
            elif "**2. Topic Analysis:**" in line:
                insights["topic_analysis"]["dominant_topics"] = {}
                topics_match = re.findall(r"\*\*Dominant Topics:\*\* (.+?) \((\d+%)\)", lines[i + 1])
                for topic, percentage in topics_match:
                    insights["topic_analysis"]["dominant_topics"][topic.lower()] = percentage

            # Mood Correlation
            elif "**3. Mood Correlation:**" in line:
                correlation_lines = []
                while i + 1 < len(lines) and lines[i + 1].startswith("*"):
                    correlation_lines.append(lines[i + 1].strip("*").strip())
                    i += 1
                insights["mood_correlation"] = " ".join(correlation_lines)

            # Visualization Metrics
            elif "**4. Metrics for Visualization:**" in line:
                visualization_metrics = {}
                while i + 1 < len(lines) and lines[i + 1].startswith("*"):
                    metric_line = lines[i + 1].strip("*").strip()
                    if ":" in metric_line:
                        key, value = metric_line.split(":", 1)
                        visualization_metrics[key.strip().lower()] = value.strip()
                    i += 1
                insights["visualization_metrics"] = visualization_metrics

            # Summary
            elif "**Overall Summary:**" in line:
                summary_lines = []
                while i + 1 < len(lines) and not lines[i + 1].startswith("**"):
                    summary_lines.append(lines[i + 1].strip())
                    i += 1
                insights["summary"] = " ".join(summary_lines)

        return insights


    def generate_journal_insights(self, entry, additional_data=None):
        insights = self.prompt_gemini_for_journal(entry, additional_data)
        return insights




if __name__ == '__main__':  # Corrected main check
    entries = [
        "I'm feeling very anxious about work deadlines. It feels overwhelming at times.",
        "Had a great day outdoors, feeling rejuvenated and calm after spending time in nature.",
        "Struggled a bit with motivation today. I felt tired and uninspired."
    ]



    topic_save_path = 'model\\topic_model'
    topic_model = AutoModelForSequenceClassification.from_pretrained(
        topic_save_path)
    topic_tokenizer = BertTokenizer.from_pretrained(topic_save_path)

    topic_model_classifier = pipeline("text-classification", model=topic_model,
                                    top_k=1, tokenizer=topic_tokenizer)


    sentiment_save_path = 'model\\sentiment_model'
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_save_path)
    sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_save_path)

    sentiment_model_classifier = pipeline("text-classification", model=sentiment_model,
                                        tokenizer=sentiment_tokenizer, top_k=None)
    

    additional_data_example = {
        "Work": {"Stress": 75, "Anxiety": 80},
        "Relaxation": {"Calm": 90, "Contentment": 85}
    }
    
    jig = JournalInsightGenerator()
    jig.one_shot_prompt()
    
    for entry in entries:
        topic_model_output = topic_model_classifier(entry)
        sentiment_model_output = sentiment_model_classifier(entry)
        threshold = 0.5
        # Flatten the nested list and filter based on score threshold
        sentiment_model_threshold = [
            entry for sublist in sentiment_model_output for entry in sublist if entry['score'] >= threshold]
        additional_data = {
            "topic_model_output": topic_model_output,
            "sentiment_model_output": sentiment_model_output
        }
        
        insights = jig.generate_journal_insights(entry, additional_data=additional_data)
        print(f"Journal Entry: {entry}")
        print(f"Journal Entry Insights: {insights}")
