# student id: 211406343
from sys import argv
from bs4 import BeautifulSoup   # will use it in order to parse xml files
import glob                     # will use it in order to get the names of files


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
        
        with open(file_name, 'r') as f:     # reading file
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
            self.sentences.append(Sentence(tokens, 'head', len(tokens)))    # adding a new sentence each loop
        
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
        with open(file_name, 'r', encoding='utf-8') as f:   # reading txt file with utf-8
            # we should not add empty lines and headers
            data = [line.strip() for line in f.readlines() if line[0] != '=' and line.strip() != ""]

        for line in data:
            words = line.split()    # splitting the line in order to get the tokens objects
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

            if count % 2 == 0 and count != 0:   # end of line
                text_ += "\n"
            else:
                text_ += " "
            count += 1
        
        with open(file_name, 'w', encoding='utf-8') as f:   # writing the final result to the output file in utf-8
            f.write(text_)


if __name__ == '__main__':

    xml_dir = argv[1]          # directory containing xml files from the BNC corpus (not a zip file)
    wiki_dir = argv[2]         # directory containing text files from Wikipedia (not a zip file)
    output_file = argv[3]

    xml_files = glob.glob(xml_dir + "/*.xml")     # a list of xml files' names
    txt_files = glob.glob(wiki_dir + "/*.txt")    # a list of txt files' names

    c = Corpus()
    
    for file in xml_files:
        c.add_xml_file_to_corpus(file)      # adding all xml files to the corpus
    
    for file in txt_files:
        c.add_text_file_to_corpus(file)     # adding all txt files to the corpus

    c.create_text_file(output_file)     # writing the output file
