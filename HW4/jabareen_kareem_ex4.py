import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, word2vec
from sys import argv
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob


# The Corpus class from previous exercises:


class Token:
    def __init__(self, t, token_word, c5, hw, pos):
        self.t = t
        self.word = token_word
        self.c5 = c5
        self.hw = hw
        self.pos = pos


class Sentence:
    def __init__(self, sentence_tokens, parent, size):
        self.tokens = sentence_tokens
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
            tokens_ = []

            for t_word in words:
                if len(t_word.contents) > 0:
                    # appending tokens to the tokens list
                    tok = Token('w', t_word.contents[0].replace(' ', ''), t_word.get('c5'), t_word.get('hw'),
                                t_word.get('pos'))

                    tokens_.append(tok)

            self.sentences.append(Sentence(tokens_, 'head', len(tokens_)))  # adding a new sentence each loop

        for sent in p_sent:
            words = sent.find_all('w')
            tokens_ = []

            for t_word in words:
                if len(t_word.contents) > 0:
                    tok = Token('w', t_word.contents[0].replace(' ', ''), t_word.get('c5'), t_word.get('hw'),
                                t_word.get('pos'))

                    tokens_.append(tok)

            self.sentences.append(Sentence(tokens_, 'p', len(tokens_)))

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is a text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as f:  # reading txt file with utf-8
            # we should not add empty lines and headers
            data = [t_line.strip() for t_line in f.readlines() if t_line[0] != '=' and t_line.strip() != ""]

        for line_ in data:
            words = line_.split()  # splitting the line in order to get the tokens objects
            tokens_ = []

            for word_ in words:
                tok = Token('w', word_.replace(' ', ''), None, None, None)
                tokens_.append(tok)

            self.sentences.append(Sentence(tokens_, 'p', len(tokens_)))

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """
        count = 0
        text_ = ""  # indicates the final result

        for sentence in self.sentences:
            tokenss = []
            for j_ in range(sentence.size):
                tokenss.append(sentence.tokens[j_].word)  # unpacking tokens

            text_ += " ".join(tokenss) + " "

            if count % 2 == 0 and count != 0:  # end of line
                text_ += "\n"
            else:
                text_ += " "
            count += 1

        with open(file_name, 'w', encoding='utf-8') as f:  # writing the final result to the output file in utf-8
            f.write(text_)

    def get_trigrams_number(self, word1, word2, word3):
        """Returns the number of appearances of the sequence of three received words in order."""
        trigrams_count = 0

        for sen in self.sentences:
            for tok_index in range(len(sen - 2)):
                if word1 == sen.tokens[tok_index].word and word2 == sen.tokens[tok_index + 1].word and \
                        word3 == sen.tokens[tok_index + 2].word:
                    trigrams_count += 1

        return trigrams_count

    def get_bigrams_number(self, word1, word2):
        """Returns the number of appearances of the sequence of two received words in order."""
        bigrams_count = 0

        for sen in self.sentences:
            for tok_index in range(len(sen - 1)):
                if word1 == sen.tokens[tok_index].word and word2 == sen.tokens[tok_index + 1].word:
                    bigrams_count += 1

        return bigrams_count


class Tweet:

    def __init__(self, sentences_=None, category=None):
        if sentences_ is None:
            self.sentences = []
        else:
            self.sentences = sentences_

        self.category = category


if __name__ == "__main__":

    # # Do the following only once!
    # # Save the GloVe text file to a word2vec file for your use:
    # glove2word2vec('glove.6B.50d.txt', 'C:\\Users\\Karee\\Documents\\NLP-Exercises\\HW4\\kv_file.kv')
    # # Load the file as KeyVectors:
    # pre_trained_model = KeyedVectors.load_word2vec_format\
    #     ('C:\\Users\\Karee\\Documents\\NLP-Exercises\\HW4\\kv_file.kv', binary=False)
    # # Save the key vectors for your use:
    # pre_trained_model.save('C:\\Users\\Karee\\Documents\\NLP-Exercises\\HW4\\kv_file.kv')
    # # Now, when handing the project, the KeyVector filename will be given as an argument.
    # # You can load it as follwing:

    pre_trained_model = KeyedVectors.load('C:\\Users\\Karee\\Documents\\NLP-Exercises\\HW4\\kv_file.kv', mmap='r')

    kv_file = argv[1]
    xml_dir = argv[2]          # directory containing xml files from the BNC corpus (not a zip file)
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]

    output_text = "Word Pairs and Distances:\n\n"

    word_pairs = [
        ('girl', 'boy'),
        ('man', 'woman'),
        ('playing', 'games'),
        ('dancing', 'love'),
        ('fast', 'slow'),
        ('sweet', 'sugar'),
        ('smoke', 'haze'),
        ('food', 'hungry'),
        ('east', 'west'),
        ('door', 'open')
    ]

    for index, word_pair in enumerate(word_pairs):
        curr_distance = KeyedVectors.distance(pre_trained_model, word_pair[0], word_pair[1])

        output_text += str(index + 1) + '. '
        output_text += word_pair[0] + " - " + word_pair[1] + " : " + str(curr_distance) + "\n"

    output_text += '\nAnalogies:\n\n'
    half_len = len(word_pairs) // 2

    for i in range(half_len):
        output_text += str(i + 1) + '. '
        output_text += word_pairs[2 * i][0] + ' : ' + word_pairs[2 * i][1] + ' , '
        output_text += word_pairs[2 * i + 1][0] + ' : ' + word_pairs[2 * i + 1][1] + '\n'

    output_text += '\nMost Similar:\n\n'

    model_results = []

    for i in range(half_len):
        output_text += str(i + 1) + '. '

        most_similar_word, _ = KeyedVectors.most_similar(pre_trained_model,
                                                         positive=[word_pairs[2 * i][1], word_pairs[2 * i + 1][0]],
                                                         negative=[word_pairs[2 * i + 1][1]])[0]

        model_results.append((word_pairs[2 * i][0], most_similar_word))

        output_text += word_pairs[2 * i][1] + ' + ' + word_pairs[2 * i + 1][0] + ' - '
        output_text += word_pairs[2 * i + 1][1] + ' = ' + most_similar_word + '\n'

    output_text += '\nDistances:\n\n'

    for i in range(half_len):
        curr_distance = KeyedVectors.distance(pre_trained_model, model_results[i][0], model_results[i][1])

        output_text += str(i + 1) + '. '
        output_text += model_results[i][0] + ' - ' + model_results[i][1] + ' : '
        output_text += str(curr_distance) + '\n'

    output_text += '\n'  # end of task 1

    # task 2 (Easy Grammy)
    # TODO: finish task 2

    xml_files = glob.glob(xml_dir + "/*.xml")  # a list of xml files' names

    main_corpus = Corpus()

    for xml_file in xml_files:
        main_corpus.add_xml_file_to_corpus(xml_file)

    lyrics_corpus = Corpus()
    lyrics_corpus.add_text_file_to_corpus(lyrics_file)  # simple tokenization since each line has exactly one sentence

    output_text += '=== New Hit ===\n\n'

    # task 3 (Tweets mapping)
    tweets_corpus = Corpus()

    tweets_list = []

    with open(tweets_file, 'r') as f_reader:
        curr_category_name = None
        for line in f_reader.readlines():
            if line[0:] == '==':  # need to find category name
                temp = line.split()
                for word in temp:
                    if '=' not in word:
                        curr_category_name = word

            else:
                temp = line.split()
                tokens = []
                sentences = []
                for word in temp:
                    if word[-1] in '!.':  # end of sentence
                        sentences.append(Sentence(tokens, 'p', len(tokens)))
                        tokens = []
                    else:
                        tokens.append(Token('w', word, None, None, None))

                if len(tokens) > 0:
                    sentences.append(Sentence(tokens, 'p', len(tokens)))

                tweets_list.append(Tweet(sentences, curr_category_name))

    # built a list of all tweets in the file after applying tokenization and saved for each tweet for what category
    # it belongs

    for tweet in tweets_list:
        tweets_corpus.sentences.extend(tweet.sentences)

    # added the sentences to the tweets corpus
    # TODO: how to get words vectors ?!
