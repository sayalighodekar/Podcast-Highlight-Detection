import nltk
from summarizer.sbert import SBertSummarizer
from semantic_chunker import SemanticChunker
from transformers import pipeline
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
nltk.download('stopwords')
nltk.download('punkt')


class HighlightDetector:
    """
    Extract most engaging parts of the podcast based on 
    a) extracting key information conveyed in the podcast through segments
    b) ranking the extracted segments based on the intensity of emotion conveyed.
    """

    def __init__(self, file_id) -> None:
        # model fo summarization task
        self.sbert_model = SBertSummarizer('paraphrase-MiniLM-L6-v2')

        # model for getting emotion-score across 7 emotions
        self.emotion_classifier = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        self.chunk_dictionary = {}
        self.file_id = file_id

    def chunk_input_data(self, data):

        # Function to call SemanticChunker, see semantic_chunker.py

        alignments = data["results"]["items"]
        transcripts = data["results"]["transcripts"][0]['transcript']

        sc = SemanticChunker()
        self.chunk_dictionary = sc.run_chunker(alignments, transcripts)

    def extract_top_sentences(self, summary_document):
        """
        Function to extract top N important "representative" sentences (extractive summary) 

        :param summary_document: document to be summarized
        :type summary_document: sring


        :return final_summary: top N sentences
        """
        # we choose N to be 5% of the total sentence length of the document
        summary = self.sbert_model(summary_document, ratio=0.05)
        final_summary = []
        for sent in nltk.sent_tokenize(summary):
            if len(tknzr.tokenize(sent)) >= 20:
                final_summary.append(sent)

        return final_summary

    def extract_potential_highlight_chunks(self, final_summary):
        """
        Function to identify segments containing representative sentences 

        :param final_summary: sentences extracted from summarization model
        :type final_summary: list


        :return potential_highlight_indices: returns indices of the segments which contain our representative sentences
        """

        potential_highlight_indices = []

        for idx, chunk in self.chunk_dictionary.items():
            for sentence in final_summary:
                if sentence in chunk["content"]:
                    potential_highlight_indices.append(idx)

        return potential_highlight_indices

    def emotion_filtering(self, potential_highlight_indices):
        """
        Function to identify segments that contain  

        :param potential_highlight_indices: indices of the engaging segments 
        :type potential_highlight_indices: list 


        :return : returns index of best segment according to our emotion based ranking model
        """

        emotion_scores = []
        for idx in potential_highlight_indices:
            current_chunk = self.chunk_dictionary[idx]

            classifer_scores = self.emotion_classifier(
                current_chunk["content"])[0]
            sorted_scores = sorted(
                classifer_scores, key=lambda x: x["score"], reverse=True)

            if sorted_scores[0]['label'] == "neutral":
                current_emotion = sorted_scores[1]
            else:

                current_emotion = sorted_scores[0]
            current_emotion["chunk_idx"] = idx
            emotion_scores.append(current_emotion)

        sorted_emotion_score = sorted(
            emotion_scores, key=lambda x: x["score"], reverse=True)
        return sorted_emotion_score[0]["chunk_idx"]

    def run_highlight_detector(self, data):
        """
        Function to run the entire highlight extraction pipeline

        :param data: transcription data 
        :type data: json  


        :return : returns best segment of the podcast, with start time, end time and content
        """

        self.chunk_input_data(data)

        summary_document = " "
        for idx, chunk in self.chunk_dictionary.items():
            summary_document = summary_document + " " + chunk["content"]

        final_summary = self.extract_top_sentences(summary_document)
        potential_highlight_indices = self.extract_potential_highlight_chunks(
            final_summary)

        chunk_idx = self.emotion_filtering(potential_highlight_indices)
        highlight_chunk = self.chunk_dictionary[chunk_idx]
        return {"file_id": self.file_id, "start_time": highlight_chunk["span"][0], "end_time": highlight_chunk["span"][1],
                "transcript_content": highlight_chunk["content"]}
