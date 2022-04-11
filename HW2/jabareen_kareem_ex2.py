# student id: 211406343
import math
from sys import argv
from bs4 import BeautifulSoup  # will use it in order to parse xml files
from glob import glob as gl  # will use it in order to get the names of files
from random import randint


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

        with open(file_name, 'r', encoding='utf-8') as jabber:  # reading file
            data = jabber.read()

        # we shall parse the xml file right now

        wtext = BeautifulSoup(data, "xml").find("wtext")

        if wtext is None:
            stext = BeautifulSoup(data, "xml").find("stext")
            heads = stext.find_all('head')
            ps = stext.find_all('p')

        else:
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
        with open(file_name, 'r', encoding='utf-8') as jabber:  # reading txt file with utf-8
            # we should not add empty lines and headers
            data = [line.strip() for line in jabber.readlines() if line[0] != '=' and line.strip() != ""]

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

        for sen in self.sentences:
            tokens = []
            for j in range(sen.size):
                tokens.append(sen.tokens[j].word)  # unpacking tokens

            text_ += " ".join(tokens) + " "

            if count % 2 == 0 and count != 0:  # end of line
                text_ += "\n"
            else:
                text_ += " "
            count += 1

        with open(file_name, 'w', encoding='utf-8') as jabber:  # writing the final result to the output file in utf-8
            jabber.write(text_)

    def get_random_sentence_length(self):
        """This method will return a random number indicating a length of a sentence"""
        rand_sen_index = randint(0, len(self.sentences) - 1)

        rand_sen = self.sentences[rand_sen_index]   # select a random sentence in the corpus then return its length
        result = len(rand_sen.tokens)

        if rand_sen.tokens[1].word == '<B>':    # beginning and ending tokens should not be counted
            result -= 1
        if rand_sen.tokens[-1].word == '<E>':
            result -= 1

        return result


class NGramModel:
    def __init__(self, n, corpus):
        self.n = n
        self.corpus = corpus
        self.vocabulary_counter = {}  # a dictionary to count each word's appearances
        self.vocabulary_size = 0  # holds the number of tokens in the corpus

        for sen in corpus.sentences:  # building the dictionary and setting the size of the vocabulary
            for tok in sen.tokens:
                self.vocabulary_size += 1
                if tok.word in self.vocabulary_counter.keys():
                    self.vocabulary_counter[tok.word] += 1
                else:
                    self.vocabulary_counter[tok.word] = 1

    def probability_in_laplace_smoothing_unigrams(self, sen: str):
        """
        This method will find the probability of a sentence depending on the object's corpus based on laplace smoothing
        for an Unigram
        :param sen: The sentence to process, we need to find the probability for it
        :return: The probability of the sentence based on laplace smoothing
        """
        if self.n != 1:
            raise ValueError("Expected a Unigram!")

        result = 0
        words = sen.split(' ')  # holds a list of words in the sentence

        last_word = words[len(words) - 1]

        if last_word[-1] == '.':  # removing the point from the last word
            words.pop(len(words) - 1)
            words.append(last_word[0: len(last_word) - 1])

        if len(words) == 0:
            raise ValueError("Empty List!")

        for word in words:
            if word in self.vocabulary_counter.keys():
                curr_tok_prob = self.vocabulary_counter[word] + 1
            else:
                curr_tok_prob = 1  # smoothing, this way we avoid giving new words the probability of 0
            curr_tok_prob /= (self.vocabulary_size + len(self.vocabulary_counter.keys()))
            result += math.log(curr_tok_prob)  # sum log instead of multiplying because the values of the probability
            # are so close to zero

        return pow(math.e, result)

    def probability_in_laplace_smoothing_bigrams(self, sen: str):
        """
        This method will find the probability of a sentence depending on the object's corpus based on laplace smoothing
        for a Bigram
        :param sen: The sentence to process, we need to find the probability for it
        :return: The probability of the sentence based on laplace smoothing
        """
        if self.n != 2:
            raise ValueError("Expected a Bigram!")

        words = sen.split(' ')  # split the sentence into a list of words

        last_word = words[len(words) - 1]

        if last_word[-1] == '.':    # remove the sentence-ending point
            words.pop(len(words) - 1)
            words.append(last_word[0: len(last_word) - 1])

        counter = 0

        for c_sen in self.corpus.sentences:
            if len(c_sen.tokens) > 0 and c_sen.tokens[0].word == words[0]:
                counter += 1

        if len(words) == 0:
            raise ValueError("Empty List!")

        result = math.log((1 + counter) / (len(self.corpus.sentences) + self.vocabulary_size))
        # initialized with the probability for the first word

        for index_ in range(len(words) - 1):
            # for every couple of words we should find the number of its appearances
            couples_counter = 0
            for c_sen in self.corpus.sentences:
                for j in range(len(c_sen.tokens) - 1):
                    if c_sen.tokens[j].word == words[index_] and c_sen.tokens[j + 1].word == words[index_ + 1]:
                        couples_counter += 1

            curr_couple_prob = couples_counter + 1
            if words[index_] in self.vocabulary_counter.keys():     # avoiding indexing error
                curr_couple_prob /= (self.vocabulary_counter[words[index_]] + self.vocabulary_size)
            else:
                curr_couple_prob /= self.vocabulary_size

            result += math.log(curr_couple_prob)    # adding log instead of multiplying since the numbers are too
            # close to 0

        # dealing with the last word:
        counter = 0
        for c_sen in self.corpus.sentences:
            if len(c_sen.tokens) > 0 and c_sen.tokens[len(c_sen.tokens) - 1].word == words[-1]:
                counter += 1

        last_word_prob = (1 + counter) / (len(self.corpus.sentences) + self.vocabulary_size)
        result += math.log(last_word_prob)

        return pow(math.e, result)

    def probability_in_linear_interpolation_trigram(self, sen: str):
        """
        This method will find the probability of a sentence depending on the object's corpus based on linear
        interpolation for a Trigram
        :param sen: The sentence to process, we need to find the probability for it
        :return: The probability of the sentence based on linear interpolation
        """
        if self.n != 3:
            raise ValueError("Expected a Trigram!")

        lam1, lam2 = 0.8, 0.15
        lam3 = 1 - lam1 - lam2  # setting parameters for the interpolation
        words = sen.split(' ')  # splitting the sentence into a list of words

        last_word = words[len(words) - 1]

        if last_word[-1] == '.':
            words.pop(len(words) - 1)
            words.append(last_word[0: len(last_word) - 1])  # removing the end point

        if len(words) == 0:
            raise ValueError("Empty List!")

        # should handle the first two words in a special way

        if words[0] in self.vocabulary_counter.keys():
            result = math.log(self.vocabulary_counter[words[0]] / self.vocabulary_size)
        else:
            result = 0

        if len(words) == 1:
            if result == 0:
                return 0

            return pow(math.e, result)

        # probability for the second word in the sentence
        eta1, eta2 = 0.65, 0.35

        counter = 0     # count the appearances of the couple of the first two words
        for sen in self.corpus.sentences:
            for index1 in range(len(sen.tokens) - 1):
                if sen.tokens[index1].word == words[0] and sen.tokens[index1 + 1].word == words[1]:
                    counter += 1

        if words[0] in self.vocabulary_counter.keys():
            result += eta1 * math.log(counter / self.vocabulary_counter[words[0]])

        if words[1] in self.vocabulary_counter.keys():
            result += eta2 * math.log(self.vocabulary_counter[words[1]] / self.vocabulary_size)

        if len(words) == 2:
            return pow(math.e, result)

        for k in range(2, len(words)):      # dealing with the rest of the sentence since we have at least 3 words,
            # so we can complete our interpolation
            if words[k] in self.vocabulary_counter.keys():  # probability of single word multiplied by lambda3
                curr_prob = lam3 * math.log(self.vocabulary_counter[words[k]] / self.vocabulary_size)
            else:
                curr_prob = 0

            counter = 0
            for sen in self.corpus.sentences:
                for _l in range(len(sen.tokens) - 1):
                    # second probability (the couples' one) multiplied by lambda2
                    if sen.tokens[_l].word == words[k - 1] and sen.tokens[_l + 1].word == words[k]:
                        counter += 1

            if words[k - 1] in self.vocabulary_counter.keys() and self.vocabulary_counter[words[k - 1]] != 0 and \
                    counter != 0:
                curr_prob += lam2 * math.log(counter / self.vocabulary_counter[words[k - 1]])

            # now we shall find the probability for the three words together

            tri_counter = 0
            bi_counter = 0

            for sen in self.corpus.sentences:
                for j in range(len(sen.tokens) - 2):

                    if sen.tokens[j].word == words[k - 2] and sen.tokens[j + 1].word == words[k - 1]:
                        bi_counter += 1
                        if sen.tokens[j + 2].word == words[k]:
                            tri_counter += 1

            if bi_counter != 0 and tri_counter != 0:    # avoid dividing by 0
                curr_prob += lam1 * math.log(tri_counter / bi_counter)

            result += curr_prob     # using logs for values very close to zero

        return pow(math.e, result)

    def generate_random_sentence_unigram(self, sen_length: int):
        """This method generates and returns a random sentence for Unigram model"""
        if self.n != 1:
            raise ValueError("Expected a Unigram model!")

        new_sen = ['<B>']
        end_token = '<E>'

        possible_words = []      # the list of all possible words for the next token

        for sen in self.corpus.sentences:
            for tok in sen.tokens:
                if tok.word != '<B>':       # we cannot add <B> to the middle\end of a sentence
                    possible_words.append(tok.word)

        for j in range(sen_length):
            rand_index = randint(0, len(possible_words) - 1)    # using random.randint to get a random index
            new_sen.append(possible_words[rand_index])

            if new_sen[-1] == end_token:
                return new_sen

        # we shall make sure we add the ending token at the end of the sentence

        if new_sen[-1] != end_token:
            new_sen.append(end_token)
        return new_sen

    def generate_random_sentence_bigram(self, sen_length: int):
        """This method generates and returns a random sentence for Bigram model"""
        if self.n != 2:
            raise ValueError("Expected a Bigram model!")

        new_sen = ['<B>']
        end_token = '<E>'

        for k in range(sen_length):
            possible_words = []

            for sen in self.corpus.sentences:
                for _i in range(len(sen.tokens) - 1):
                    if sen.tokens[_i].word == new_sen[-1]:
                        possible_words.append(sen.tokens[_i + 1].word)

            # now possible_words has all the words that appeared after the last word we have added (with replications)

            rand_index = randint(0, len(possible_words) - 1)
            new_sen.append(possible_words[rand_index])

            if new_sen[-1] == end_token:
                return new_sen

        # make sure we add <E> at the end of the sentence

        if new_sen[-1] != end_token:
            new_sen.append(end_token)
        return new_sen

    def generate_random_sentence_trigram(self, sen_length: int):
        """This method generates and returns a random sentence for Trigram model"""
        if self.n != 3:
            raise ValueError("Expected a Trigram model!")

        new_sen = ['<B>']
        end_token = '<E>'
        # adding the first non-special token should be handled in a different way since we still do not have enough data

        possible_words = []
        for sen in self.corpus.sentences:
            possible_words.append(sen.tokens[1].word)  # the beginning token appears only at the beginning of each
            # sentence that is why we are taking the second token

        rand_index = randint(0, len(possible_words) - 1)
        new_sen.append(possible_words[rand_index])

        for _i in range(sen_length - 1):  # already added a token
            possible_words = []

            for _sen in self.corpus.sentences:
                for _j in range(len(_sen.tokens) - 2):
                    if _sen.tokens[_j].word == new_sen[-2] and _sen.tokens[_j + 1].word == new_sen[-1]:
                        possible_words.append(_sen.tokens[_j + 2].word)

            rand_index = randint(0, len(possible_words) - 1)
            new_sen.append(possible_words[rand_index])

            if new_sen[-1] == end_token:
                return new_sen

        if new_sen[-1] != end_token:
            new_sen.append(end_token)
        return new_sen

    def generate_random_sentence(self, sen_length: int):
        """This method generates and returns a random sentence based on the value of `n`"""
        if sen_length < 0:
            raise ValueError("Expected a non-negative sentence length.")

        if sen_length == 0:
            return ''

        # the sentence can end before reaching the random length received as a parameter
        self.vocabulary_counter['<E>'] = len(self.corpus.sentences)

        if self.n == 1:
            return self.generate_random_sentence_unigram(sen_length)
        if self.n == 2:
            return self.generate_random_sentence_bigram(sen_length)
        if self.n == 3:
            return self.generate_random_sentence_trigram(sen_length)

        raise ValueError("Unsupported Model!")


if __name__ == '__main__':

    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    output_text = ""    # this text will be written to the output file

    # the following list contains the sentences for task 1
    sentences = [
        "May the Force be with you.",
        "I’m going to make him an offer he can’t refuse.",
        "Ogres are like onions.",
        "You’re tearing me apart, Lisa!",
        "I live my life one quarter at a time."
    ]

    xml_files = gl(xml_dir + "/*.xml")      # a list of strings indicating all xml files' names
    c = Corpus()

    for file in xml_files:
        c.add_xml_file_to_corpus(file)      # adding xml files to the corpus and build a language model

    unigrams = NGramModel(n=1, corpus=c)
    bigrams = NGramModel(n=2, corpus=c)
    trigrams = NGramModel(n=3, corpus=c)

    # the output for the firs task:

    output_text += "*** Sentence Predictions ***\n\n"

    for model in (unigrams, bigrams, trigrams):
        if model.n == 1:
            output_text += "Unigrams Model:\n\n"
        elif model.n == 2:
            output_text += "Bigrams Model:\n\n"
        else:
            output_text += "Trigrams Model:\n\n"

        for sentence in sentences:
            output_text += sentence + "\n"
            output_text += "Probability: "
            # we shall convert the log of the probability to string and add it to output_text
            if model.n == 3:
                output_text += str(math.log(model.probability_in_linear_interpolation_trigram(sen=sentence)))
            elif model.n == 2:
                output_text += str(math.log(model.probability_in_laplace_smoothing_bigrams(sen=sentence)))
            else:
                output_text += str(math.log(model.probability_in_laplace_smoothing_unigrams(sen=sentence)))

            output_text += '\n\n'

    # second task

    # special tokens
    beginning_token = Token('w', '<B>', None, None, None)
    ending_token = Token('w', '<E>', None, None, None)

    for sentence in c.sentences:    # adding the special tokens to each and every sentence in the corpus
        temp = [beginning_token]
        temp.extend(sentence.tokens)
        temp.append(ending_token)

        sentence.tokens = temp
    unigrams.corpus = bigrams.corpus = trigrams.corpus = c      # updating the models' corpus attribute

    output_text += "*** Random Sentence Generation ***\n\n"

    for model in (unigrams, bigrams, trigrams):
        if model.n == 1:
            output_text += "Unigrams Model:\n\n"
        elif model.n == 2:
            output_text += "\nBigrams Model:\n\n"
        else:
            output_text += "\nTrigrams Model:\n\n"

        for i in range(5):
            random_length = model.corpus.get_random_sentence_length()       # get a random length for the next sentence
            new_sentence = model.generate_random_sentence(sen_length=random_length)     # generate a new sentence

            for index in range(1, len(new_sentence) - 1):
                output_text += new_sentence[index]
                if index == len(new_sentence) - 2:
                    output_text += '.'      # adding end of line point at the end of each sentence
                else:
                    output_text += ' '      # adding spaces between words in the same sentence

            output_text += '\n'

    with open(output_file, 'w', encoding='utf-8') as f:     # finally, write to the output file in utf-8
        f.write(output_text)
