from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
from bs4 import BeautifulSoup
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
        # etc = "[.][.]"

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


## Do the following only once!

## Save the GloVe text file to a word2vec file for your use:
# glove2word2vec(<downloaded_text_filename>, <full_path_vector_filename>')
## Load the file as KeyVectors:
# pre_trained_model = KeyedVectors.load_word2vec_format(<full_path_vector_filename.kv>, binary=False)
## Save the key vectors for your use:
# pre_trained_model.save(<full_path_keyvector_filename.kv>)

## Now, when handing the project, the KeyVector filename will be given as an argument.
## You can load it as follwing:
# pre_trained_model = KeyedVectors.load(<full_path_keyvector_filename>.kv, mmap='r')


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
    corpus.add_text_file_to_corpus(lyrics_file)

    output_text = ""



