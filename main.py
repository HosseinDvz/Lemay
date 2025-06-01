from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import WebsiteClassifier

app = FastAPI()
clf = WebsiteClassifier()

class ClassificationRequest(BaseModel):
    website: str
    content: str

@app.post("/classify")
def classify(request: ClassificationRequest):
    try:
        label = clf.classify_row(request.website, request.content)
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
