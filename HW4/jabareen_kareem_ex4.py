import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import random


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
            tok_s = []

            for t_word in words:
                # appending tokens to the tokens list
                if len(t_word.contents) > 0:
                    tok = Token('w', t_word.contents[0].replace(' ', ''), t_word.get('c5'), t_word.get('hw'),
                                t_word.get('pos'))

                    tok_s.append(tok)
            self.sentences.append(Sentence(tok_s, 'head', len(tok_s)))  # adding a new sentence each loop

        for sent in p_sent:
            words = sent.find_all('w')
            tok_s = []

            for t_word in words:
                if len(t_word.contents) > 0:
                    tok = Token('w', t_word.contents[0].replace(' ', ''), t_word.get('c5'), t_word.get('hw'),
                                t_word.get('pos'))

                    tok_s.append(tok)
            self.sentences.append(Sentence(tok_s, 'p', len(tok_s)))

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
            tokens_s = []
            for j_ in range(sentence.size):
                tokens_s.append(sentence.tokens[j_].word)  # unpacking tokens

            text_ += " ".join(tokens_s) + " "

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
            for tok_index in range(len(sen.tokens) - 2):
                if word1 == sen.tokens[tok_index].word and word2 == sen.tokens[tok_index + 1].word and \
                        word3 == sen.tokens[tok_index + 2].word:
                    trigrams_count += 1

        return trigrams_count

    def get_bigrams_number(self, word1, word2):
        """Returns the number of appearances of the sequence of two received words in order."""
        bigrams_count = 0

        for sen in self.sentences:
            for tok_index in range(len(sen.tokens) - 1):
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


def get_new_song(song_corpus, words_to_swap, xml_corpus, kv_trained_model):
    """
    Creates a new song by changing a word per line in the song contained in `song_corpus`.
    :param song_corpus: Corpus having only the grammy winner song tokenized.
    :param words_to_swap: A list of words, each word is from a separate line in the grammy winner song to swap with a
    similar word from given corpus.
    :param xml_corpus: The corpus to search for appearances for similar words in and to add them to the output song.
    :param kv_trained_model: Trained KeyedVector for finding similarity between words.
    :return: Corpus object containing the new song tokenized.
    :raise ValueError if the length of `words_to_swap` is unequal to the number of sentences in `song_corpus`, or if
    got at least one word that does not exist in the sentence to change.
    """

    if len(song_corpus.sentences) != len(words_to_swap):
        raise ValueError("The number of words to swap must be equal to the number of sentences in the corpus.")

    new_song_corpus = Corpus()

    for sen_ind, song_sentence in enumerate(song_corpus.sentences):
        replaced_at_least_once = False
        newest_tokens = []

        for token_index, song_token in enumerate(song_sentence.tokens):
            temp_song_word = song_token.word

            # a token could end\start with more than one punctuation like ending with ?"
            while len(temp_song_word) > 0:
                if temp_song_word[-1] in '.,?!")':
                    temp_song_word = temp_song_word[:len(temp_song_word) - 1]
                    continue
                break

            while len(temp_song_word) > 0:
                if temp_song_word[0] in '.,"(?!':
                    temp_song_word = temp_song_word[1:]
                    continue
                break

            if '-' in temp_song_word:
                end_index = 1
                for char_index, char in enumerate(temp_song_word):
                    if char == '-':
                        end_index = char_index
                        break

                temp_song_word = temp_song_word[:end_index]

            if temp_song_word.lower() == words_to_swap[sen_ind].lower():
                most_similar_10 = KeyedVectors.most_similar(kv_trained_model, positive=[temp_song_word.lower()])
                trigrams_counters = []
                bigrams_counters = []

                for similar_word, _ in most_similar_10:
                    if len(song_sentence.tokens) - 1 > token_index > 0:
                        # word to change is in the middle of the sentence
                        trigrams_counters.append(xml_corpus.get_trigrams_number(
                            song_sentence.tokens[token_index - 1].word, similar_word,
                            song_sentence.tokens[token_index + 1].word))

                    elif token_index == 0:  # word to change is at the beginning of the sentence
                        trigrams_counters.append(xml_corpus.get_bigrams_number(similar_word,
                                                                               song_sentence.tokens[1].word))

                    else:  # word to change is the last one of the sentence
                        trigrams_counters.append(xml_corpus.get_bigrams_number(song_sentence.tokens[-2].word,
                                                                               similar_word))

                if any(trigrams_counters):  # at least one counter is greater than 0
                    max_counter, max_similarity, res_word = max(trigrams_counters), 0, None

                    # if several words got an equal and positive value for the counter, we should choose the one most
                    # similar to the original word

                    for ind, count in enumerate(trigrams_counters):
                        if count == max_counter and most_similar_10[ind][1] >= max_similarity:
                            res_word = most_similar_10[ind][0]

                    next_token = Token('w', res_word, None, None, None)

                else:  # all counters are 0, we should check bigrams
                    for similar_word, _ in most_similar_10:
                        if len(song_sentence.tokens) - 1 > token_index > 0:
                            # word to change is in the middle of the sentence

                            temp1 = xml_corpus.get_bigrams_number(song_sentence.tokens[token_index - 1].word,
                                                                  similar_word)

                            temp2 = xml_corpus.get_bigrams_number(similar_word,
                                                                  song_sentence.tokens[token_index + 1].word)
                            bigrams_counters.append(temp1 + temp2)

                        elif token_index == 0:  # word to change is at the beginning of the sentence
                            bigrams_counters.append(xml_corpus.get_bigrams_number(similar_word,
                                                                                  song_sentence.tokens[1].word))

                        else:  # word to change is the last one of the sentence
                            bigrams_counters.append(xml_corpus.get_bigrams_number(song_sentence.tokens[-2].word,
                                                                                  similar_word))

                    if any(bigrams_counters):
                        max_counter, max_similarity, res_word = max(bigrams_counters), 0, None

                        for ind, count in enumerate(bigrams_counters):
                            if count == max_counter and most_similar_10[ind][1] >= max_similarity:
                                res_word = most_similar_10[ind][0]

                        next_token = Token('w', res_word, None, None, None)

                    else:   # all counters are 0 again, now we should just choose the most similar word
                        most_similar_10.sort(key=lambda x: x[1],  reverse=True)
                        next_token = Token('w', most_similar_10[0][0], None, None, None)

                replaced_at_least_once = True
            else:
                next_token = Token('w', song_token.word, None, None, None)

            newest_tokens.append(next_token)

        if not replaced_at_least_once:
            error_str = f'Word {words_to_swap[sen_ind]} not found. Sentence index = {sen_ind}.'
            with open('ErrorLog.txt', 'w', encoding='utf-8') as error_f:
                error_f.write(error_str)

            raise ValueError("Word not found, open ErrorLog text file for more info!")

        new_song_corpus.sentences.append(Sentence(newest_tokens, 'p', len(newest_tokens)))

    return new_song_corpus


def get_corpus_as_text(corpus):
    """
    Creates a text based on the parameter `corpus`.
    :param corpus: Corpus object to create text from.
    :return: The text generated from the corpus.
    """

    generated_text = ""

    for corpus_sentence in corpus.sentences:
        for ind in range(len(corpus_sentence.tokens)):
            generated_text += corpus_sentence.tokens[ind].word

            if ind == len(corpus_sentence.tokens) - 1:
                generated_text += '\n'
            else:
                generated_text += ' '

    return generated_text


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
    xml_dir = argv[2]  # directory containing xml files from the BNC corpus (not a zip file)
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

    xml_files = glob.glob(xml_dir + "/*.xml")  # a list of xml files' names

    main_corpus = Corpus()

    for xml_file in xml_files:
        main_corpus.add_xml_file_to_corpus(xml_file)

    lyrics_corpus = Corpus()
    lyrics_corpus.add_text_file_to_corpus(lyrics_file)  # simple tokenization since each line has exactly one sentence

    output_text += '=== New Hit ===\n\n'

    words_to_change = ['baby', 'doing', 'at', 'Oh', 'trap', 'robe', 'alone', 'warm', 'like', 'dancing', 'and',
                       'mansion', 'games', 'coming', 'lay', 'leave', 'leave', 'leave', 'leave', 'way', 'tonight',
                       'coming', 'sweet', 'bite', 'Purple', 'filets', 'keep', 'love', 'talking', 'bathtub', 'jump',
                       'games', 'coming', 'lay', 'leave', 'leave', 'leave', 'leave', 'way', 'tonight', 'coming',
                       'need', 'gotta', 'tryna', 'ah', 'leave', 'leave', 'hoping', 'way', 'want', 'coming',
                       'la', 'coming', 'woo', 'woo', 'la', 'coming', 'oh', 'gotta', 'waiting', 'coming',
                       'waiting', 'adore', 'la']

    my_new_song = get_new_song(lyrics_corpus, words_to_change, main_corpus, pre_trained_model)

    output_text += get_corpus_as_text(my_new_song) + '\n'

    # task 3 (Tweets mapping)
    tweets_corpus = Corpus()

    tweets_list = []

    with open(tweets_file, 'r', encoding='utf-8') as f_reader:
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

    arithmetic_tweets_vectors = []
    random_tweets_vectors = []
    custom_tweets_vectors = []

    weak_words = ['in', 'on', 'at', 'the', 'of', 'his', 'her', 'their', 'there', 'my', 'our', 'i', 'me', 'ours', 'hers',
                  'you', 'your', 'we', 'he', 'she', 'them', 'off']

    weak_words_weight = 0.1

    # added the sentences to the tweets corpus

    for sentence_index, tweet_sentence in enumerate(tweets_corpus.sentences):
        words_vectors = ([], [])

        for tweet_token in tweet_sentence.tokens:
            temp_word = tweet_token.word

            while len(temp_word) > 0:
                if not temp_word[-1].isalpha():
                    temp_word = temp_word[:len(temp_word) - 1]
                    continue
                break

            while len(temp_word) > 0:
                if not temp_word[0].isalpha():
                    temp_word = temp_word[1:]
                    continue
                break

            words_vectors[0].append(KeyedVectors.get_vector(pre_trained_model, temp_word))
            words_vectors[1].append(temp_word)      # saving the word itself for creating our custom vectors

        arithmetic_tweets_vectors.append(sum(words_vectors[0]) / len(words_vectors[0]))

        temp_sum = None

        for i in range(len(words_vectors[0])):
            random_weight = random.uniform(0, 10)

            while random_weight == 0 or random_weight == 10:
                random_weight = random.uniform(0, 10)
            if i == 0:
                temp_sum = words_vectors[0][sentence_index][i] * random_weight
            else:
                temp_sum += words_vectors[0][sentence_index][i] * random_weight

        temp_sum /= len(words_vectors[0])
        random_tweets_vectors.append(temp_sum)

        for i in range(len(words_vectors[0])):
            if words_vectors[1][i] in weak_words:
                random_weight = weak_words_weight
            else:
                random_weight = random.uniform(3, 10)

            if i == 0:
                temp_sum = words_vectors[0][sentence_index][i] * random_weight
            else:
                temp_sum += words_vectors[0][sentence_index][i] * random_weight

        temp_sum /= len(words_vectors[0])
        custom_tweets_vectors.append(temp_sum)

    arithmetic_tweets_vectors = np.array(arithmetic_tweets_vectors)
    random_tweets_vectors = np.array(random_tweets_vectors)
    custom_tweets_vectors = np.array(custom_tweets_vectors)

    pca = PCA()
    pca.fit(arithmetic_tweets_vectors)
    arithmetic_results = pca.fit_transform(arithmetic_tweets_vectors)

    plt.scatter(arithmetic_results[:, 1], arithmetic_results[:, 2])
    plt.show()

    pca = PCA()
    pca.fit(random_tweets_vectors)
    random_results = pca.fit_transform(random_tweets_vectors)

    plt.scatter(random_results[:, 1], random_results[:, 2])
    plt.show()

    pca = PCA()
    pca.fit(custom_tweets_vectors)
    custom_results = pca.fit_transform(custom_tweets_vectors)

    plt.scatter(custom_results[:, 1], custom_results[:, 2])
    plt.show()
