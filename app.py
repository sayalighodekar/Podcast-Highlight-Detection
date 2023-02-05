
# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
import uvicorn
import json
# Declaring our FastAPI instance
app = FastAPI()

# Defining path operation for root endpoint


@app.get('/')
def main():
    return {'message': 'Hello! This is a Podcast Highlight Retreival Service! Go to "http://127.0.0.1:8000/docs" for UI excecution'}

# An endpoint that accepts transcription file (JSON) and returns highlights


@app.post("/generate_highlights/")
async def detect_highlights(file: UploadFile):

    data = json.loads(file.file.read())

    from HighLightDetector import HighlightDetector
    extractor = HighlightDetector(file.filename,)
    highlights = extractor.run_highlight_detector(data)

    return highlights
