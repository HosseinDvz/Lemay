import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm

class WebsiteClassifier:
    def __init__(self, labels=None):
        self.labels = labels or [
            "educational", "news", "entertainment", "shopping",
            "social media", "health", "finance", "sports"
        ]
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\b(403|404|500|302)\b.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def classify_row(self, website: str, content: str) -> str:
        clean_site = self.clean_text(website)
        clean_text = self.clean_text(content)
        combined = f"{clean_site}: {clean_text}"
        if not clean_text:
            return "unknown"
        result = self.classifier(combined[:1000], self.labels)  # Truncate to stay within model limit
        return result["labels"][0]

    def add_label_column(self, input_csv: str, output_csv: str):
        df = pd.read_csv(input_csv)
        tqdm.pandas(desc="Classifying")
        df["predicted_label"] = df.progress_apply(
            lambda row: self.classify_row(row["website"], row["content"]), axis=1
        )
        df.to_csv(output_csv, index=False)
        print(f"✅ Output saved to {output_csv}")


if __name__ == "__main__":
    input_path = "chunk_1_results.csv"
    output_path = "chunk_1_labeled.csv"
    clf = WebsiteClassifier()
    clf.add_label_column(input_path, output_path)
