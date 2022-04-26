# student id: 211406343

import random
from sys import argv
import numpy as np
from bs4 import BeautifulSoup  # will use it in order to parse xml files
import glob  # will use it in order to get the names of files
import gender_guesser.detector as gen
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class Token:
    def __init__(self, t, word, c5, hw, pos):
        self.t = t
        self.word = word
        self.c5 = c5
        self.hw = hw
        self.pos = pos


class Sentence:
    def __init__(self, tokens, parent, size, gender='unknown'):
        self.tokens = tokens
        self.parent = parent
        self.size = size
        self.gender = gender


class Corpus:
    def __init__(self):
        self.sentences = []
        self.chunks = []

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
        writer_gender = None

        if wtext is None:  # the file is stext which has no author name
            stext = BeautifulSoup(data, "xml").find("stext")
            heads = stext.find_all('head')
            ps = stext.find_all('p')

        else:
            writer_full_name = BeautifulSoup(data, "xml").find('bncDoc').find('teiHeader').find('fileDesc')
            writer_full_name = writer_full_name.find('sourceDesc').find('bibl')

            temp = writer_full_name.find_all('author')

            if temp is not None and len(temp) == 1:  # there is exactly one writer
                d = gen.Detector()
                writer_full_name = writer_full_name.find('author').text
                writer_full_name = writer_full_name.split(' ')
                writer_gender = d.get_gender(writer_full_name[-1])

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

        last_chunk = []  # we consider chunks from paragraphs only

        for sent in p_sent:
            words = sent.find_all('w')
            tokens = []

            for word in words:
                if len(word.contents) > 0:
                    tok = Token('w', word.contents[0].replace(' ', ''), word.get('c5'), word.get('hw'), word.get('pos'))
                    tokens.append(tok)
            self.sentences.append(Sentence(tokens, 'p', len(tokens)))

            if writer_gender is not None:
                last_chunk.append(Sentence(tokens, 'head', len(tokens)))

                if len(last_chunk) == 10:
                    self.chunks.append((last_chunk, writer_gender))
                    last_chunk = []

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is a text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as f:  # reading txt file with utf-8
            # we should not add empty lines and headers
            data = [line.strip() for line in f.readlines() if line[0] != '=' and line.strip() != ""]

        for line in data:
            words = line.split()  # splitting the line in order to get the tokens objects
            tokens = []

            for word in words:
                tok = Token('w', word.replace(' ', ''), None, None, None)
                tokens.append(tok)
            self.sentences.append(Sentence(tokens, 'p', len(tokens)))

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


class Classify:

    def __init__(self, corpus):
        self.corpus = corpus
        self.chunk_bows = []
        self.custom_vectors = []

    def down_sample(self):
        male_chunks, female_chunks = [], []

        for ch in self.corpus.chunks:
            if ch[1] == u'male' or ch[1] == 'male':
                male_chunks.append(ch)
            elif ch[1] == u'female' or ch[1] == 'female':
                female_chunks.append(ch)

        if len(male_chunks) == len(female_chunks):
            return

        chunks_to_drop = abs(len(male_chunks) - len(female_chunks))

        if len(male_chunks) > len(female_chunks):
            for sam in random.sample(male_chunks, chunks_to_drop):
                male_chunks.remove(sam)
        else:
            for sam in random.sample(female_chunks, chunks_to_drop):
                female_chunks.remove(sam)

        self.corpus.chunks = []

        for x in male_chunks:
            self.corpus.chunks.append(x)
        for x in female_chunks:
            self.corpus.chunks.append(x)

    def create_bows(self):
        temp = []

        for chunk_ in self.corpus.chunks:
            chunk_text = ''
            for sen_ in chunk_[0]:
                for tok in sen_.tokens:
                    chunk_text += tok.word + ' '

            temp.append(chunk_text)

        cv = CountVectorizer()
        return cv.fit_transform(temp)

    def create_custom_vectors(self):
        self.custom_vectors = []

        for chunk_ in self.corpus.chunks:
            sens_len = []
            for i in range(10):
                curr_len = len(chunk_[0][i].tokens)
                sens_len.append(curr_len)

            self.custom_vectors.append(sens_len)

        self.custom_vectors = np.array(self.custom_vectors)     # converting a list of lists to a 2D matrix

    def get_counters(self):
        if len(self.corpus.chunks) == 0:
            return 0, 0

        male_counter, female_counter = 0, 0

        for ch in self.corpus.chunks:
            if ch[1] == 'male' or ch[1] == u'male':
                male_counter += 1
            elif ch[1] == 'female' or ch[1] == u'female':
                female_counter += 1

        return female_counter, male_counter


if __name__ == '__main__':

    # xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    # output_file = argv[2]  # output file name, full path

    xml_dir = 'XML_files'
    output_file = 'output.txt'

    xml_files = glob.glob(xml_dir + "/*.xml")  # a list of xml files' names

    corp = Corpus()

    for file in xml_files:
        corp.add_xml_file_to_corpus(file)  # adding all xml files to the corpus

    classify = Classify(corpus=corp)

    output_text = ""

    f_chunks_counter, m_chunks_counter = classify.get_counters()

    output_text += "Before Down-Sampling:\n"
    output_text += "Female " + str(f_chunks_counter) + "\tMale: " + str(m_chunks_counter) + "\n\n"

    classify.down_sample()
    f_chunks_counter, m_chunks_counter = classify.get_counters()

    output_text += "After Down-Sampling:\n"
    output_text += "Female " + str(f_chunks_counter) + "\tMale: " + str(m_chunks_counter) + "\n\n"

    # classifying ..

    # k_folds = KFold(n_splits=10)
    knn = KNeighborsClassifier()
    classify.create_bows()
    classify.create_custom_vectors()

    y_vector = []
    for chunk in classify.corpus.chunks:
        y_vector.append(chunk[1])

    # for train_index, test_index in k_folds.split(X=classify.chunk_bows, y=y_vector):     # cross validation algorithm
    #     knn.fit(classify.chunk_bows[train_index], y_vector[train_index])
    #     y_predictions = knn.predict(y_vector[test_index])

    scores = cross_val_score(knn, classify.chunk_bows, y_vector)

    output_text += "== BoW Classification ==\n"
    output_text += "Cross Validation Accuracy: "
    output_text += str(scores) + '\n\n'

    # split into test and train algorithm
    x_train, x_test, y_train, y_test = train_test_split(classify.chunk_bows, y_vector, test_size=0.3)

    knn.fit(x_train, y_train)
    y_predictions = knn.predict(x_test)

    output_text += "Splitting into train and test sets Accuracy: "
    output_text += str(accuracy_score(y_test, y_predictions))       # output accuracy for splitting into train and test
    output_text += '\n'

    scores = cross_val_score(knn, classify.custom_vectors, y_vector)

    output_text += "\n== Custom Feature Vector Classification ==\n"
    output_text += "Cross Validation Accuracy: "
    output_text += str(scores) + '\n\n'

    # split into test and train algorithm
    x_train, x_test, y_train, y_test = train_test_split(classify.custom_vectors, y_vector, test_size=0.3)

    knn.fit(x_train, y_train)
    y_predictions = knn.predict(x_test)

    output_text += "Splitting into train and test sets Accuracy: "
    output_text += str(accuracy_score(y_test, y_predictions))  # output accuracy for splitting into train and test
    output_text += '\n'

    with open(output_file, 'w', encoding='utf-8') as jabber:
        jabber.write(output_text)
