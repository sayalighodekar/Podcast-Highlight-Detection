# Podcast-Highlight-Detection 🎙️🎧
A ML-based POC for extracting highlights (most interesting) parts of PodCast using Semantic segmentation, Summarization and Emotion scores.

In this repository we extract small clips (like a reel), which represent the most exciting and informative parts of the podcast. These clips can be used to promote the podcast or give a preview of the podcast.

## Approach

For a podcast segment to be exciting to want to focus on the two following properties
- **Information** conveyed in the segment should be important to the overall "theme" of the podcast.
- **Emotional Intensity** expressed in the clip creates engaging anf viral content.

The first step is to split the transcript into meaningful segments according to the topics spoken. We do this by using sub-topic drift detection metric implemented in the 

