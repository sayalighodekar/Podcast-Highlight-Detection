# Podcast-Highlight-Detection 🎙️🎧
A ML-based POC for extracting highlights (most interesting) parts of PodCast using Semantic segmentation, Summarization and Emotion scores.

In this repository we extract small clips (like a reel), which represent the most exciting and informative parts of the podcast. These clips can be used to promote the podcast or give a preview of the podcast.

## Usage

To run the Highlight Detection module
```python 
from HighLightDetector import HighlightDetector

file_id = open('Transcript file you want to extract highlights from',"r)
data = json.loads(file_id.read())

extractor = HighlightDetector(file_id)
highlights = extractor.run_highlight_detector(data)
```

Input `file_id` contains the JSON file extracted after calling Google ASR API for transcription. 
Check a sample file at   

## Approach

For a podcast segment to be exciting to want to focus on the two following properties
- **Information** conveyed in the segment should be important to the overall "theme" of the podcast.
- **Emotional Intensity** expressed in the clip creates engaging anf viral content.

The workflow is as follows:

1.  **Topic-Based Text Segmentation** The first step is to split the transcript into meaningful segments according to the topics spoken. We do this by using sub-topic drift detection metrics as implemented in the [Texttiling Algorithm](https://aclanthology.org/J97-1003/). This is a unsupervised approach which uses lexical cohesion between sentence blocks to determine topic boundaries. 

2.  **Extract top segments using Extractive Summarization** Now that we have segmented the data into chunks of different topics, we need to identify important chunks. We do this by running an extractive summarization module to capture top 5% sentences which capture the essence of the document. The extractive summarizer uses clutering and ranking methods along with **Sentence embeddings**. In this way we capture the salient information from the transcript. We then consider topic segments which contain these top-ranked sentences. This also reduces the search space for our next step. 
For implementation we use [Sentence-Bert Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer#use-sbert) with ``paraphrase-MiniLM-L6-v2`` pretrained model as it runs very fast.  

**NOTE 1**: We can also use [Keyword extraction ](https://github.com/MaartenGr/KeyBERT) methods. But the problem with this method is that keywords can be spread all over the document. So even if we perform semantic matching of keyword phrases with topic segments, its not guranteed that the segment with most keywords will be most informative/engaging. 

**NOTE 2**: Through my experiments I found that generating top 5% sentences was giving best results, however we can tune this parameter based on memory and latency requirements.

3.  **Ranking segments based on emotional intensity** Now that we have potential "highlight" segments, how do we choose the most engaging one? I decided to use emotion-based ranking for this. We obtain emotion-scores across all 6 emotions (anger, disgust, fear, joy, sadness, surprise) for each segments, and select the segment displaying highest emotion-score. We choose the highest score across all the segments (and across all emotions).
