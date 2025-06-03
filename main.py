from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import WebsiteClassifier

app = FastAPI()

# Initialize the website classifier instance once at startup
clf = WebsiteClassifier()

class ClassificationRequest(BaseModel):
    """
    Request body schema for classification.
    Contains the website name and its corresponding textual content.
    """
    website: str
    content: str

@app.post("/classify")
def classify(request: ClassificationRequest):
    """
    Classify the content of a website using a zero-shot classification model.

    Parameters:
    - request: JSON body with 'website' and 'content' fields.

    Returns:
    - A dictionary containing the predicted label.

    Raises:
    - HTTPException with status code 500 if classification fails.
    """
    try:
        label = clf.classify_row(request.website, request.content)
        return {"label": label}
    except Exception as e:
        # Return internal server error with the exception message
        raise HTTPException(status_code=500, detail=str(e))

