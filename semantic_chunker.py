import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TextTilingTokenizer

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = TweetTokenizer()


class SemanticChunker:
    """
    Segment documents into topic chunks based on sub-topic drift metrics. 
    """

    def __init__(self) -> None:
        pass

    def align_text(self, aligments, sentences):
        """
        Function to align sentences given start and end timings at word level.

        :param alignments: alignments at word-level (from Google API). Eg:
                            {'start_time': 0.32,
                            'end_time': 0.4,
                            'alternatives': [{'confidence': '1.0', 'content': 'Hi'}],
                            'type': 'pronunciation'}
        :type aligments: dict


        :param sentences: sentences extracted from transcript
        :type sentences: list 

        :return final_segments: dictionary containing sentence start_time, end_time and text
        """

        final_segments = []
        counter = 0
        for idx, s in enumerate(sentences):

            start = counter
            end = start + len(tokenizer.tokenize(s))

            if "end_time" not in aligments[end-2].keys():
                segment = {"start": aligments[start]["start_time"],
                           "end": aligments[end-1]["end_time"], "text": s}
                counter = end - 1

            else:

                segment = {"start": aligments[start]["start_time"],
                           "end": aligments[end-2]["end_time"], "text": s}
                counter = end

            final_segments.append(segment)

        return final_segments

    def create_tiles(self, input_sentences, tokenizer):
        """
        Function to segment input sentences into "semantic tiles"

        :param input_sentences: list of sentences to be split.

        :param tokenizer: Tokenizer 

        :return tiles: text segments split based on topic-shifts
        """
        temp_text = "\n\n\t".join(input_sentences)
        tokens = tokenizer.tokenize(temp_text)
        tiles = []
        for token in tokens:
            tiles.append(token)

        return tiles

    def extract_timings(self, final_segments):

        timings = []
        for segment in final_segments:
            timings.append([segment["start"], segment["end"]])

        return timings

    def start_end_time(self, tiles, timings):
        """
        Function for finding start and end time for aligning tiles with timings.
        :param tiles: input text segments.
        :param timings: tuple(start_time, end_time) of utterances.
        :return: tiles with corresponding timings.
        """

        tiles_time = []
        start_flag = 0
        end_flag = 0
        for t in tiles:
            lines = t.split("\n\n\t")
            lines = [i for i in lines if i]
            end_flag = start_flag + (len(lines) - 1)
            tiles_time.append([timings[start_flag][0], timings[end_flag][1]])
            start_flag = end_flag + 1

        return tiles_time

    def create_chunk_dictionary(self, timings_aligned_with_tiles, tiles):
        chunk_dictionary = {}
        counter = 0

        # Usually first two tiles contain "intros" and the last two contain "outros", which we do not want to include in our "highlight"
        tiles = tiles[2:]
        tiles = tiles[:-2]

        timings_aligned_with_tiles = timings_aligned_with_tiles[2:]
        timings_aligned_with_tiles = timings_aligned_with_tiles[:-2]

        for x, y in zip(timings_aligned_with_tiles, tiles):
            y = y.replace("\n\n\t", " ").lstrip()
            chunk_dictionary[counter] = {"span": x, "content": y}
            counter += 1
        return chunk_dictionary

    def run_chunker(self, alignments, transcript):
        """
        Function for run the pipeline to segment text into chunks based on sub-topic shifts

        :param alignments: alignments at word-level (from Google API). Eg:
                            {'start_time': 0.32,
                            'end_time': 0.4,
                            'alternatives': [{'confidence': '1.0', 'content': 'Hi'}],
                            'type': 'pronunciation'}

        :param transcript: string containing transcript 

        :return: "chunk_dictionary" (A dictionary contaning segment text, its start and end time)
        """

        sentences = nltk.sent_tokenize(transcript)
        final_segments = self.align_text(alignments, sentences)

        texttiling_tokenizer = TextTilingTokenizer(
            w=int(0.1 * len(final_segments)), k=10)
        tiles = self.create_tiles(sentences, texttiling_tokenizer)
        timings = self.extract_timings(final_segments)

        timings_aligned_with_tiles = self.start_end_time(tiles, timings)
        chunk_dictionary = self.create_chunk_dictionary(
            timings_aligned_with_tiles, tiles)

        return chunk_dictionary
