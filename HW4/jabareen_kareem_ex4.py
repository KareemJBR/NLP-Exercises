from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import word2vec
from sys import argv
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import re
import os

# The Corpus class from previous exercises:


class Token:
    def __init__(self, t, word, c5, hw, pos):
        self.t = t
        self.word = word
        self.c5 = c5
        self.hw = hw
        self.pos = pos


class Sentence:
    def __init__(self, tokens, parent, size):
        self.tokens = tokens
        self.parent = parent
        self.size = size


class Corpus:
    def __init__(self):
        self.sentences = []

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """

        with open(file_name, 'r') as f:  # reading file
            data = f.read()

        # we shall parse the xml file right now

        wtext = BeautifulSoup(data, "xml").find("wtext")

        heads = wtext.find_all('head')
        ps = wtext.find_all('p')

        head_sent = []
        for h in heads:
            head_sent += h.find_all('s')

        p_sent = []
        for p in ps:
            p_sent += p.find_all('s')

        for sent in head_sent:
            words = sent.find_all('w')
            tokens = []

            for word in words:
                # appending tokens to the tokens list
                tok = Token('w', word.contents[0].replace(' ', ''), word.get('c5'), word.get('hw'), word.get('pos'))
                tokens.append(tok)
            self.sentences.append(Sentence(tokens, 'head', len(tokens)))  # adding a new sentence each loop

        for sent in p_sent:
            words = sent.find_all('w')
            tokens = []

            for word in words:
                tok = Token('w', word.contents[0].replace(' ', ''), word.get('c5'), word.get('hw'), word.get('pos'))
                tokens.append(tok)
            self.sentences.append(Sentence(tokens, 'p', len(tokens)))

    # def add_text_file_to_corpus(self, file_name: str):
    #     """
    #     This method will receive a file name, such that the file is a text file (from Wikipedia), read the content
    #     from it and add it to the corpus in the manner explained in the exercise instructions.
    #     :param file_name: The name of the text file that will be read
    #     :return: None
    #     """
    #     with open(file_name, 'r', encoding='utf-8') as f:  # reading txt file with utf-8
    #         # we should not add empty lines and headers
    #         data = [line.strip() for line in f.readlines() if line[0] != '=' and line.strip() != ""]
    #
    #     start_index = 0
    #
    #     for line in data:
    #         for i_ in range(len(line)):
    #             if line[i_] in '!.\n':
    #
    #
    #         words = line.split()  # splitting the line in order to get the tokens objects
    #         tokens = []
    #
    #         for word in words:
    #             tok = Token('w', word.replace(' ', ''), None, None, None)
    #             tokens.append(tok)
    #         self.sentences.append(Sentence(tokens, 'p', len(tokens)))

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is a text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """
        with open(os.path.join(os.getcwd(), file_name), 'r', encoding='utf8') as f:
            data = f.read()

        alphabets = "([A-Za-z])"
        numbers = "([0-9])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He/s|She/s|It/s|They/s|Their/s|Our/s|We/s|But/s|However/s|That/s|This/s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"

        data = re.sub(prefixes, "\\1<dont_splt>", data)
        data = re.sub(websites, "<dont_splt>\\1", data)
        data = re.sub(" " + alphabets + "[.] ", " \\1<dont_splt> ", data)  # examples (W. Bush, H.W ...)
        data = re.sub(acronyms + " " + starters, "\\1<dont_splt> \\2", data)
        data = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
                      "\\1<dont_splt>\\2<dont_splt>\\3<dont_splt>", data)
        data = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets,
                      "\\1<dont_splt>\\2<dont_splt>\\3<dont_splt>", data)
        data = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<dont_splt>\\2", data)  # examples (U. S., H. W.)
        data = re.sub(numbers + "[.]" + numbers, "\\1<dont_splt>\\2", data)
        data = re.sub(" " + suffixes + "[.]", " \\1<dont_splt>", data)

        if "”" in data:
            data = data.replace(".”", "”.")
        if "\"" in data:
            data = data.replace(".\"", "\".")
        if ")" in data:
            data = data.replace(".)", ").")
        if "!" in data:
            data = data.replace("!\"", "\"!")
        if "?" in data:
            data = data.replace("?\"", "\"?")

        data = data.replace(".", f'.\n')
        data = data.replace("?", f'?\n')
        data = data.replace("!", f'!\n')
        data = data.replace("<dont_splt>", ".")

        # Removing extra new lines
        data = re.sub(r'\n+', '\n', data)
        # Removing double spaces
        data = re.sub(r' +', ' ', data)
        # Remove empty titles
        while re.search('==\n==', data):
            data = re.sub(r'==+.*?==+\n=', r'=', data)
        # Take care of an empty title at the end of the string
        data = re.sub(r'==+.*?==+$', '', data)

        data = data.splitlines()  # list of sentences
        sentences_list = []
        for line in data:
            if len(line) > 0:
                line = line.split()  # list of tokens
                tokens = []

                for token in line:
                    token_obj = Token('w', token, None, None, None)
                    tokens.append(token_obj)

                sentence_obj = Sentence(tokens, 'p', len(tokens))
                sentences_list.append(sentence_obj)

        for sen in sentences_list:
            self.sentences.append(sen)

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """
        count = 0
        text_ = ""  # indicates the final result

        for sentence in self.sentences:
            tokens = []
            for i in range(sentence.size):
                tokens.append(sentence.tokens[i].word)  # unpacking tokens

            text_ += " ".join(tokens) + " "

            if count % 2 == 0 and count != 0:  # end of line
                text_ += "\n"
            else:
                text_ += " "
            count += 1

        with open(file_name, 'w', encoding='utf-8') as f:  # writing the final result to the output file in utf-8
            f.write(text_)

    def change_word(self, word):
        pass


if __name__ == "__main__":

    kv_file = argv[1]
    xml_dir = argv[2]          # directory containing xml files from the BNC corpus (not a zip file)
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]

    xml_files = glob.glob(xml_dir + "/*.xml")  # a list of xml files' names

    corpus = Corpus()

    for xml_file in xml_files:
        corpus.add_xml_file_to_corpus(xml_file)

    output_text = "Word Pairs and Distances:\n\n"

    word_pairs = [
        ('kissing', 'cuddling'),
        ('adore', 'love'),
        ('playing', 'games'),
        ('dancing', 'romancing'),
        ('smooth', 'newborn'),
        ('sweet', 'tight'),
        ('smoke', 'haze'),
        ('filetes', 'hungry'),
        ('east', 'west'),
        ('door', 'open')
    ]

    for index, word_pair in enumerate(word_pairs):
        curr_distance = 0   # TODO: calculate distance between word_pair[0] and word_pair[1]
        output_text += str(index + 1)
        output_text += word_pair[0] + " - " + word_pair[1] + " : " + str(curr_distance) + "\n"

    output_text += '\nAnalogies:\n\n'
    half_len = len(word_pairs) // 2

    for i in range(half_len):
        output_text += str(i + 1) + '. '
        output_text += word_pairs[2*i][0] + ' : ' + word_pairs[2*i][1] + ' , '
        output_text += word_pairs[2*i+1][0] + ' : ' + word_pairs[2*i+1][1] + '\n'

    output_text += '\nMost Similar:\n\n'
    word2vec_model = word2vec.Word2Vec()
    word2vec_model.load(kv_file)

    model_results = []

    for i in range(half_len):
        output_text += str(i + 1) + '. '
        most_similar_word = word2vec_model.most_similar(positive=[word_pairs[2*i][1], word_pairs[2*i+1][0]],
                                                        negative=[word_pairs[2*i+1][1]])

        model_results.append((word_pairs[2*i][0], most_similar_word))

        output_text += word_pairs[2*i][1] + ' + ' + word_pairs[2*i+1][0] + ' - ' + word_pairs[2*i+1][1] + ' = '
        output_text += most_similar_word + '\n'

    output_text += '\nDistances:\n\n'

    for i in range(half_len):
        curr_distance = 0   # TODO: calculate distance between model_results[i][0] and model_results[i][1]
        output_text += str(i + 1) + '. '
        output_text += model_results[i][0] + ' - ' + model_results[i][1] + ' : '
        output_text += str(curr_distance) + '\n'

    output_text += '\n'     # end of task 1

    # task 2 (Easy Grammy)
    # TODO: finish task 2
    grammy_corpus = Corpus()
    grammy_corpus.add_text_file_to_corpus(lyrics_file)

    output_text += '=== New Hit ===\n\n'

    # task 3 (Tweets mapping)

    tweets_corpus = Corpus()
    tweets_corpus.add_text_file_to_corpus(tweets_file)
