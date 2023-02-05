# Podcast-Highlight-Detection 🎙️🎧
A ML-based POC for extracting highlights (most interesting) parts of PodCast using Semantic segmentation, Summarization and Emotion scores.

In this repository we extract small clips (like a reel), which represent the most exciting and informative parts of the podcast. These clips can be used to promote the podcast or give a preview of the podcast.

## Usage
### To run locally 
To run the Highlight Detection module
```python 
from HighLightDetector import HighlightDetector

file_id = open('Transcript file you want to extract highlights from',"r)
data = json.loads(file_id.read())

extractor = HighlightDetector(file_id)
highlights = extractor.run_highlight_detector(data)
```

Input `file_id` contains the JSON file extracted after calling Google ASR API for transcription. 
Check a sample file format at  `transcripts/`. 

Output for `transcripts/radiolab_dinopocalypse-redux (1).json` Link to the podcast (https://radiolab.org/episodes/dinopocalypse-redux)
```json
{
  "file_id": "radiolab_dinopocalypse-redux (1).json",
  "start_time": 2276.56,
  "end_time": 2392.4,
  "transcript_content": "You might feel the earth rumble there's some shaking and then that rumbling, that shaking comes with it, a big wave from the Sea Whoa, and so you get this big push that comes in and then what happens is that comes in is you're already starting to get the glass balls from the heavens and so and what they see. Is You get this like wave of kind of what seems to be almost like raining glass balls and then that's like mixed in with the mud from like the title search and the layers of things that are dying and the fish like some of the details that stand out to me, the most are. The fish are all generally pointed in the same direction and they're like stacked. Pretty tightly mouths open and their fins splayed, but one of the things I think is super cool. Is that all that different stuff? We talked about happening across the globe in our original show, like it probably got really hot, like you know that was Jay Malo. She was like it's really hot dog Robertson was talking about like the boiler, the boiler effect, and then we talked about that flash of blue light, and we talked about things raining from the sky and we talked about June or July. All that stuff. A lot of that stuff was based on really smart models. This seems to be a place that actually will provide evidence either for or against those models like charred tree trunks, which I think made like j Milosch really happy because he was like. I did get really hot you know, and then they were like the fish wrapped around trees, and then there appears to be a dinosaur bone and possibly a dinosaur bone with skin still attached and Kirk Johnson said, if, if that is, if it really is a dinosaur bone and that site is connected to the asteroid, impact like they think it is, it would be the youngest dinosaur ever."
}
```
### To run the WEBAPI
To run the server using `fastapi` and `univcorn`

```python3
uvicorn app:app --reload
```
This will run the app on `http://127.0.0.1:8000/`
To run the interactive API `http://127.0.0.1:8000/docs/`

Sample output for `a16z_developers-as-creatives (1).json`:
![Screen Shot 2023-02-05 at 2 52 58 AM](https://user-images.githubusercontent.com/52694032/216807865-b72562e9-f1c0-4637-a4f5-2346b8e9b204.png)



## Approach

For a podcast segment to be exciting to want to focus on the two following properties
- **Information** conveyed in the segment should be important to the overall "theme" of the podcast.
- **Emotional Intensity** expressed in the clip creates engaging anf viral content.

The workflow is as follows:

### 1. Topic-Based Text Segmentation
 The first step is to split the transcript into meaningful segments according to the topics spoken. We do this by using sub-topic drift detection metrics as implemented in the [Texttiling Algorithm](https://aclanthology.org/J97-1003/). This is a unsupervised approach which uses lexical cohesion between sentence blocks to determine topic boundaries. 

**NOTE** The efficiency tiling algorithm is highly dependant on the pseudosentence size (w parameter). For this model, I have set this size as 10% of the document length. For smaller documents, this parameter as not very optimal as it produces extremely small segments. 
We can also use [Elbow methods](https://en.wikipedia.org/wiki/Elbow_method_(clustering)#:~:text=In%20cluster%20analysis%2C%20the%20elbow,number%20of%20clusters%20to%20use.) to find the optimal size, however this will be a tradeoff between accuracy and speed. See 'Suggested Improvements' section for an alternative to texttiling.

### 2.Extract top segments using Extractive Summarization
 Now that we have segmented the data into chunks of different topics, we need to identify important chunks. We do this by running an extractive summarization module to capture top 5% sentences which capture the essence of the document. The extractive summarizer uses clutering and ranking methods along with **Sentence embeddings**. In this way we capture the salient information from the transcript. We then consider topic segments which contain these top-ranked sentences. This also reduces the search space for our next step. 
For implementation we use [Sentence-Bert Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer#use-sbert) with ``paraphrase-MiniLM-L6-v2`` pretrained model as it runs very fast.  

**NOTE 1**: We can also use [Keyword extraction ](https://github.com/MaartenGr/KeyBERT) methods. But the problem with this method is that keywords can be spread all over the document. So even if we perform semantic matching of keyword phrases with topic segments, its not guranteed that the segment with most keywords will be most informative/engaging. 

**NOTE 2**: Through my experiments I found that generating top 5% sentences was giving best results, however we can tune this parameter based on memory and latency requirements.

### 3. Ranking segments based on emotional intensity
 Now that we have potential "highlight" segments, how do we choose the most engaging one? This [paper](https://arxiv.org/pdf/1503.04723.pdf) explains the corelation of high-arousal emotions and virality of content. I decided to use emotion-based ranking for this. We obtain emotion-scores across all 6 emotions (anger, disgust, fear, joy, sadness, surprise) for each segments, and select the segment displaying highest emotion-score. We choose the highest score across all the segments (and across all emotions).

## TODOS and suggested Further Improvements
- **Speech-based emotion recognition** can be more useful as it categories emotions based on pitch/ sound intensity and other acoustic paramters.
- **[Neural models for topical change detection]https://huggingface.co/dennlinger/roberta-cls-consec)** While texttiling is a very fast unsupervised appraoach to find topic changes, we can further improve its accuracy by predicting coherence between two sequences. Especially if the detected highlight is very small, we can use the above method to check cohesion with next segment.
