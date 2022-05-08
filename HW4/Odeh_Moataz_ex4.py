from gensim.scripts.glove2word2vec import glove2word2vec
from sys import argv
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the GolVe to vector files, blacken after loading (or do it in a separate Python file):
# glove2word2vec(<downloaded text file_300>, <C:\Users\mo3ta\Desktop\Hw4NLP>)
# glove2word2vec(<downloaded text file_50>, <C:\Users\mo3ta\Desktop\Hw4NLP>)

# Implement your code here #


def NewTweet(tweet, word, corpus):
    most_similar_vector = pre_trained_model_50.most_similar(positive=[word], topn=10)
    my_unigrams, my_bigrams, my_trigrams = Arrange(corpus)

    MaxNumOfOccurunces = 0
    j = 0
    i = 0
    temp_flag = 0
    chosen_word = ''
    before = ''
    after = ''

    splited = tweet.split()

    for token in splited:
        if token == word:
            if i == 0:
                temp = (splited[0], splited[1], splited[2])
                temp_flag = 0

            elif i == (len(splited) - 1):
                temp = (splited[i - 2], splited[i - 1], splited[i])
                temp_flag = 1

            elif 0 < i < (len(splited) - 1):
                temp = (splited[i - 1], splited[i], splited[i + 1])
                temp_flag = 2

        else:
            i += 1
            continue

        for n in range(9):
            temp_similar_word = most_similar_vector[j][0]

            if temp_flag == 0:

                if len(splited) > 2:

                    temp = (temp_similar_word, splited[1], splited[2])
                    j += 1

                    if temp in my_trigrams:
                        if my_trigrams[temp] > MaxNumOfOccurunces:

                            MaxNumOfOccurunces = my_trigrams[temp]

                            chosen_word = ' '.join(temp)

                            before = ''

                            after = ' '.join(splited[3:len(splited) + 1])

                        else:
                            MaxNumOfOccurunces = MaxNumOfOccurunces

                    elif temp not in my_trigrams or MaxNumOfOccurunces == 0:
                        BigramsNumOfOccurunces = 0

                        temp_1 = (temp_similar_word, splited[1])

                        if temp_1 in my_bigrams:
                            BigramsNumOfOccurunces += my_bigrams[temp_1]

                        if BigramsNumOfOccurunces > MaxNumOfOccurunces:

                            MaxNumOfOccurunces = BigramsNumOfOccurunces
                            chosen_word = ' '.join(temp_1)
                            before = ''
                            after = ' '.join(splited[2:len(splited) + 1])

                        else:
                            MaxNumOfOccurunces = MaxNumOfOccurunces

                    elif (temp_1 not in my_bigrams) or (MaxNumOfOccurunces == 0):
                        chosen_word = ' '.join(most_similar_vector[0][0])

                    else:

                        continue

            if temp_flag == 2:
                if len(splited) > 2 and i > 0 and (i < len(splited) - 1):

                    temp = (splited[i - 1], temp_similar_word, splited[i + 1])

                    j += 1

                    if temp in my_trigrams:
                        if my_trigrams[temp] > MaxNumOfOccurunces:

                            MaxNumOfOccurunces = my_trigrams[temp]
                            chosen_word = ' '.join(temp)
                            before = ' '.join(splited[0:i - 1])
                            after = ' '.join(splited[i + 2:len(splited) + 1])

                        else:
                            MaxNumOfOccurunces = MaxNumOfOccurunces

                    elif temp not in my_trigrams or MaxNumOfOccurunces == 0:

                        BigramsNumOfOccurunces = 0

                        temp_1 = (temp_similar_word, splited[i + 1])
                        temp_2 = (splited[i - 1], temp_similar_word)

                        if temp_1 in my_bigrams:
                            BigramsNumOfOccurunces += my_bigrams[temp_1]

                        if temp_2 in my_bigrams:
                            BigramsNumOfOccurunces += my_bigrams[temp_2]

                        if BigramsNumOfOccurunces > MaxNumOfOccurunces:

                            MaxNumOfOccurunces = BigramsNumOfOccurunces
                            chosen_word = ' '.join(temp_1)
                            before = ' '.join(splited[0:i])
                            after = ' '.join(splited[i + 2:len(splited) + 1])

                        else:
                            MaxNumOfOccurunces = MaxNumOfOccurunces

                    elif (temp_1 not in my_bigrams and temp_2 not in my_bigrams) or MaxNumOfOccurunces == 0:
                        chosen_word = ' '.join(most_similar_vector[0][0])

                    else:
                        continue

            elif temp_flag == 1:

                temp = (splited[i - 2], splited[i - 1], temp_similar_word)

                j += 1

                if temp in my_trigrams:
                    if my_trigrams[temp] >= MaxNumOfOccurunces:

                        MaxNumOfOccurunces = my_trigrams[temp]
                        chosen_word = ' '.join(temp)
                        before = ' '.join(splited[0:i - 2])
                        after = ' '.join(splited[i + 1:len(splited) + 1])

                    else:
                        MaxNumOfOccurunces = MaxNumOfOccurunces

                elif temp not in my_trigrams or MaxNumOfOccurunces == 0:

                    BigramsNumOfOccurunces = 0

                    temp_2 = (splited[i - 1], temp_similar_word)

                    if temp_2 in my_bigrams:
                        BigramsNumOfOccurunces = my_bigrams[temp_2]

                    if BigramsNumOfOccurunces >= MaxNumOfOccurunces:

                        MaxNumOfOccurunces = BigramsNumOfOccurunces
                        chosen_word = ' '.join(temp_2)
                        before = ' '.join(splited[0:i - 1])
                        after = ' '.join(splited[i + 1:len(splited) + 1])

                    else:
                        MaxNumOfOccurunces = MaxNumOfOccurunces

                elif (temp_2 not in my_bigrams or (MaxNumOfOccurunces == 0)):
                    chosen_word = ' '.join(most_similar_vector[0][0])

                else:
                    continue

    returned_tweet = str(before) + ' ' + str(chosen_word) + ' ' + str(after)

    return returned_tweet


def Arrange(file):
    my_list = []  # This will contain the tokens
    unigrams = {}  # for unigrams and their counts
    bigrams = {}  # for bigrams and their counts
    trigrams = {}  # for trigrams and their counts.
    # 1. Read the textfile, split it into a list

    # en_list = en_text.strip().split()
    my_list = file.strip().split()

    # 2. Generate unigrams frequencies : 
    for item in my_list:
        if not item in unigrams:
            unigrams[item] = 1
        else:
            unigrams[item] += 1

    # 3. Generate bigrams with frequencies : 
    # def CreateBigrams(corpus):
    for i in range(len(my_list) - 1):

        temp = (my_list[i], my_list[i + 1])

        if not temp in bigrams:
            bigrams[temp] = 1

        else:
            bigrams[temp] += 1

    for j in range(len(my_list) - 2):

        temp = (my_list[j], my_list[j + 1], my_list[j + 2])

        if not temp in trigrams:
            trigrams[temp] = 1

        else:
            trigrams[temp] += 1

    return unigrams, bigrams, trigrams


def ConstantFunction(model, song_words, size):
    number_of_words = 0

    my_vector = np.zeros(50, dtype=int)
    avg_vector = np.zeros(50, dtype=int)

    for word in song_words:

        word = word.lower()

        if word in model:

            try:

                number_of_words += 1

                my_vector = my_vector + (np.array(model.get_vector(word))) * 1

            except:

                my_vector = my_vector

        avg_vector = [num / number_of_words for num in my_vector]

    return avg_vector


def RandomFunction(model, song_words, size):
    number_of_words = 0

    my_vector = np.zeros(50, dtype=int)
    avg_vector = np.zeros(50, dtype=int)

    for word in song_words:

        word = word.lower()

        if word in model:

            try:

                number_of_words += 1
                rand_number = np.random.uniform(size=1, low=0.1, high=5)
                my_vector = my_vector + (np.array(model.get_vector(word))) * rand_number

            except:
                my_vector = my_vector

        avg_vector = [num / number_of_words for num in my_vector]

    return avg_vector


def MyWeightsFunction(model, song_words, size):
    number_of_words = 0

    my_vector = np.zeros(50, dtype=int)
    avg_vector = np.zeros(50, dtype=int)

    for word in song_words:

        word = word.lower()

        if word in model:

            try:

                number_of_words += 1

                my_vector = my_vector + (np.array(model.get_vector(word))) * 0.7

            except:

                my_vector = my_vector

        avg_vector = [num / number_of_words for num in my_vector]

    return avg_vector


if __name__ == "__main__":
    en_corpus = argv[1]  # English corpus, text file
    glove_file_50 = argv[2]  # 50-vector, vec file
    glove_file_300 = argv[3]  # 300-vector, vec file
    song_lyrics = argv[4]  # Lyrics, text file
    output_file = argv[5]  # Output, text file

    # Building the models:
    # pre_trained_model_50 = KeyedVectors.load_word2vec_format(glove_file_50, binary=False)
    # pre_trained_model_300 = KeyedVectors.load_word2vec_format(glove_file_300, binary=False)

    pre_trained_model_50 = KeyedVectors.load(glove_file_50, mmap='r')
    pre_trained_model_300 = KeyedVectors.load(glove_file_300, mmap='r')

    # Biden's Tweets:
    tweets = [
        "America, I'm honored that you have chosen me to lead our great country. The work ahead of us will be hard, but I promise you this: I will be a President for all Americans — whether you voted for me or not. I will keep the faith that you have placed in me.",
        "If we act now on the American Jobs Plan, in 50 years, people will look back and say this was the moment that America won the future.",
        "Gun violence in this country is an epidemic — and it’s long past time Congress take action. It matters whether you continue to wear a mask. It matters whether you continue to socially distance. It matters whether you wash your hands. It all matters and can help save lives.",
        "If there’s one message I want to cut through to everyone in this country, it’s this: The vaccines are safe. For yourself, your family, your community, our country — take the vaccine when it’s your turn and available. That’s how we’ll beat this pandemic.",
        "Today, America is officially back in the Paris Climate Agreement. Let’s get to work.",
        "Today, in a bipartisan vote, the House voted to impeach and hold President Trump accountable. Now, the process continues to the Senate—and I hope they’ll deal with their Constitutional responsibilities on impeachment while also working on the other urgent business of this nation.",
        "The work of the next four years must be the restoration of democracy and the recovery of respect for the rule of law, and the renewal of a politics that’s about solving problems — not stoking the flames of hate and chaos.",
        "America is so much better than what we’re seeing today.",
        "Here’s my promise to you: I’ll be a president for all Americans. Whether you voted for me or not, I’ll wake up every single morning and work to make your life better.",
        "We can save 60,000-100,000 lives in the weeks and months ahead if we step up together. Wear a mask. Stay socially distanced. Avoid large indoor gatherings. Each of us has a duty to do what we can to protect ourselves, our families, and our fellow Americans."]

    # Implement your program here #

    with open(en_corpus, "r+", encoding='utf-8') as English_corpus:
        my_corpus = English_corpus.read()

    with open(glove_file_50, "r+", encoding='utf-8') as glove_50:

        with open(glove_file_300, "r+", encoding='utf-8') as glove_300:

            with open(song_lyrics, "r+", encoding='utf-8') as songs:

                my_songs = songs.readlines()

                with open(output_file, "w+", encoding='utf-8') as myout:

                    myout.write("-*-*-*-\n")
                    myout.write("\n")
                    myout.write("\n")
                    myout.write("=== < 50 > Word Model === \n")
                    myout.write("\n")
                    myout.write(" \n Word Pairs and Distances: (50 Word Model) \n")
                    myout.write("\n")

                    ################# 1st pair ##################
                    word1 = "win"
                    word2 = "lose"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("1. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " > \n")

                    ################# 2nd pair ##################

                    word1 = "flower"
                    word2 = "door"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("2. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 3rd pair ##################

                    word1 = "fast"
                    word2 = "slow"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("3. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 4th pair ##################

                    word1 = "anger"
                    word2 = "rage"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("4. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 5th pair ##################

                    word1 = "admit"
                    word2 = "confess"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("5. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " > \n")

                    ################# 6th pair ##################

                    word1 = "ancient"
                    word2 = "old"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("6. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 7th pair ##################

                    word1 = "book"
                    word2 = "read"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("7. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 8th pair ##################

                    word1 = "wrong"
                    word2 = "right"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("8. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 9th pair ##################

                    word1 = "female"
                    word2 = "male"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("9. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 10th pair ##################

                    word1 = "swim"
                    word2 = "pool"
                    distance = pre_trained_model_50.similarity(word1, word2)
                    myout.write("10. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################################ Part a.2 #####################################

                    myout.write("\n Analogies: \n")

                    myout.write("1. < foot > : < football > , < hand > : < handball > \n")
                    myout.write("2. < movie > : < watch > , < coffee > : < drink > \n")
                    myout.write("3. < fly > : < airplane > , < ride > : < bus > \n")
                    myout.write("4. < tall > : < short > , < white > : < black > \n")
                    myout.write("5. < water > : < drink > , < food > : < eat > \n")

                    ################# Most Similar #####################

                    myout.write(" \n Most Similar: \n")

                    ############# 1st expression #######################

                    returned_word1 = pre_trained_model_50.most_similar(positive=['football', 'handball'],
                                                                       negative=['hand'])
                    myout.write("1. < foot = football + handball - hand > = < " + str(returned_word1[0][0]) + " > \n")

                    ############# 2nd expression #######################

                    returned_word2 = pre_trained_model_50.most_similar(positive=['watch', 'drink'], negative=['coffee'])
                    myout.write("2. < movie = watch + drink - coffee > = < " + str(returned_word2[0][0]) + " > \n")

                    ############# 3rd expression #######################

                    returned_word3 = pre_trained_model_50.most_similar(positive=['airplane', 'bus'], negative=['ride'])
                    myout.write("3. < fly = airplane + bus - ride > = < " + str(returned_word3[0][0]) + " > \n")

                    ############# 4th expression #######################

                    returned_word4 = pre_trained_model_50.most_similar(positive=['short', 'black'], negative=['white'])
                    myout.write("4. < tall = short + black - white > = < " + str(returned_word4[0][0]) + " > \n")

                    ############# 5th expression #######################

                    returned_word5 = pre_trained_model_50.most_similar(positive=['drink', 'eat'], negative=['food'])
                    myout.write("5. < water = drink + eat - food > = < " + str(returned_word5[0][0]) + " > \n")

                    ############################### Distances #########################################

                    myout.write(" \n Distances: \n")

                    ##################### 1st Distance #########################

                    myout.write("1. < foot > - < " + str(returned_word1[0][0]) + " > : < " + str(
                        pre_trained_model_50.similarity("foot", str(returned_word1[0][0]))) + " > \n")

                    ##################### 2nd Distance #########################

                    myout.write("2. < movie > - < " + str(returned_word2[0][0]) + " > : < " + str(
                        pre_trained_model_50.similarity("movie", returned_word2[0][0])) + " > \n")

                    ##################### 3rd Distance #########################

                    myout.write("3. < fly > - < " + str(returned_word3[0][0]) + " > : < " + str(
                        pre_trained_model_50.similarity("fly", returned_word3[0][0])) + " > \n")

                    ##################### 4th Distance #########################

                    myout.write("4. < tall > - < " + str(returned_word4[0][0]) + " > : < " + str(
                        pre_trained_model_50.similarity("tall", returned_word4[0][0])) + " > \n")

                    ##################### 5th Distance #########################

                    myout.write("5. < water > - < " + str(returned_word5[0][0]) + " > : < " + str(
                        pre_trained_model_50.similarity("water", returned_word5[0][0])) + " > \n")

                    ####################### 300 Word Model #######################################

                    ################ 300 model #####################################

                    myout.write("\n === < 300 > Word Model === \n")
                    myout.write("\n Word Pairs and Distances: (300 Word Model) \n")
                    myout.write("\n")

                    ################# 1st pair ##################
                    word1 = "win"
                    word2 = "lose"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("1. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " > \n")

                    ################# 2nd pair ##################

                    word1 = "flower"
                    word2 = "door"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("2. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 3rd pair ##################

                    word1 = "fast"
                    word2 = "slow"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("3. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 4th pair ##################

                    word1 = "anger"
                    word2 = "rage"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("4. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 5th pair ##################

                    word1 = "admit"
                    word2 = "confess"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("5. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " > \n")

                    ################# 6th pair ##################

                    word1 = "ancient"
                    word2 = "old"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("6. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 7th pair ##################

                    word1 = "book"
                    word2 = "read"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("7. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 8th pair ##################

                    word1 = "wrong"
                    word2 = "right"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("8. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 9th pair ##################

                    word1 = "female"
                    word2 = "male"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("9. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################# 10th pair ##################

                    word1 = "swim"
                    word2 = "pool"
                    distance = pre_trained_model_300.similarity(word1, word2)
                    myout.write("10. < " + str(word1) + " > - <" + str(word2) + " > : < " + str(distance) + " >\n")

                    ################################ Part a.2 #####################################

                    myout.write("\n Analogies: \n")

                    myout.write("1. < foot > : < football > , < hand > : < handball > \n")
                    myout.write("2. < movie > : < watch > , < coffee > : < drink > \n")
                    myout.write("3. < fly > : < airplane > , < ride > : < bus > \n")
                    myout.write("4. < tall > : < short > , < white > : < black > \n")
                    myout.write("5. < water > : < drink > , < food > : < eat > \n")

                    ################# Most Similar #####################

                    myout.write(" \n Most Similar: \n")

                    ############# 1st expression #######################

                    returned_word1 = pre_trained_model_300.most_similar(positive=['football', 'handball'],
                                                                        negative=['hand'])
                    myout.write("1. < foot = football + handball - hand > = < " + str(returned_word1[0][0]) + " > \n")

                    ############# 2nd expression #######################

                    returned_word2 = pre_trained_model_300.most_similar(positive=['watch', 'drink'],
                                                                        negative=['coffee'])
                    myout.write("2. < movie = watch + drink - coffee > = < " + str(returned_word2[0][0]) + " > \n")

                    ############# 3rd expression #######################

                    returned_word3 = pre_trained_model_300.most_similar(positive=['airplane', 'bus'], negative=['ride'])
                    myout.write("3. < fly = airplane + bus - ride > = < " + str(returned_word3[0][0]) + " > \n")

                    ############# 4th expression #######################

                    returned_word4 = pre_trained_model_300.most_similar(positive=['short', 'black'], negative=['white'])
                    myout.write("4. < tall = short + black - white > = < " + str(returned_word4[0][0]) + " > \n")

                    ############# 5th expression #######################

                    returned_word5 = pre_trained_model_300.most_similar(positive=['drink', 'eat'], negative=['food'])
                    myout.write("5. < water = drink + eat - food > = < " + str(returned_word5[0][0]) + " > \n")

                    ############################### Distances #########################################

                    myout.write(" \n Distances: \n")

                    ##################### 1st Distance #########################

                    myout.write("1. < foot > - < " + str(returned_word1[0][0]) + " > : < " + str(
                        pre_trained_model_300.similarity("foot", returned_word1[0][0])) + " > \n")

                    ##################### 2nd Distance #########################

                    myout.write("2. < movie > - < " + str(returned_word2[0][0]) + " > : < " + str(
                        pre_trained_model_300.similarity("movie", returned_word2[0][0])) + " > \n")

                    ##################### 3rd Distance #########################

                    myout.write("3. < fly > - < " + str(returned_word3[0][0]) + " > : < " + str(
                        pre_trained_model_300.similarity("fly", returned_word3[0][0])) + " > \n")

                    ##################### 4th Distance #########################

                    myout.write("4. < tall > - < " + str(returned_word4[0][0]) + " > : < " + str(
                        pre_trained_model_300.similarity("tall", returned_word4[0][0])) + " > \n")

                    ##################### 5th Distance #########################

                    myout.write("5. < water > - < " + str(returned_word5[0][0]) + " > : < " + str(
                        pre_trained_model_300.similarity("water", returned_word5[0][0])) + " > \n")

                    #################################### part 2 ##################################################################

                    words_to_change = ['honored', 'work', 'faith', 'act', 'violence', 'mask', 'socially', 'wash',
                                       'lives', 'message', 'vaccine', 'pandemic', 'officially', 'get', 'impeach',
                                       'business', 'four', 'better', 'president', 'single', 'ahead', 'mask',
                                       'distanced', 'gatherings', 'duty']
                    k = 0

                    myout.write("-*-*-*-\n")
                    myout.write("\n")
                    myout.write("\n")
                    myout.write("=== New Tweets === \n")

                    myout.write("\n")
                    myout.write("\n")

                    tweet1 = [
                        "America, I'm honored that you have chosen me to lead our great country. The work ahead of us will be hard, but I promise you this: I will be a President for all Americans — whether you voted for me or not. I will keep the faith that you have placed in me."]

                    myout.write("1.")

                    my_tweet = str(tweet1).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet2 = [
                        "If we act now on the American Jobs Plan, in 50 years, people will look back and say this was the moment that America won the future."]

                    myout.write("2.")

                    my_tweet = str(tweet2).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet3 = [
                        "Gun violence in this country is an epidemic — and it’s long past time Congress take action. It matters whether you continue to wear a mask. It matters whether you continue to socially distance. It matters whether you wash your hands. It all matters and can help save lives."]

                    myout.write("3.")

                    my_tweet = str(tweet3).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet4 = [
                        "If there’s one message I want to cut through to everyone in this country, it’s this: The vaccines are safe. For yourself, your family, your community, our country — take the vaccine when it’s your turn and available. That’s how we’ll beat this pandemic."]

                    myout.write("4.")

                    my_tweet = str(tweet4).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet5 = ["Today, America is officially back in the Paris Climate Agreement. Let’s get to work."]

                    myout.write("5.")

                    my_tweet = str(tweet5).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet6 = [
                        "Today, in a bipartisan vote, the House voted to impeach and hold President Trump accountable. Now, the process continues to the Senate—and I hope they’ll deal with their Constitutional responsibilities on impeachment while also working on the other urgent business of this nation."]

                    myout.write("6.")

                    my_tweet = str(tweet6).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet7 = [
                        "The work of the next four years must be the restoration of democracy and the recovery of respect for the rule of law, and the renewal of a politics that’s about solving problems — not stoking the flames of hate and chaos."]

                    myout.write("7.")

                    my_tweet = str(tweet7).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet8 = ["America is so much better than what we’re seeing today."]

                    myout.write("8.")

                    my_tweet = str(tweet8).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet9 = [
                        "Here’s my promise to you: I’ll be a president for all Americans. Whether you voted for me or not, I’ll wake up every single morning and work to make your life better."]

                    myout.write("9.")

                    my_tweet = str(tweet9).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    tweet10 = [
                        "We can save 60,000-100,000 lives in the weeks and months ahead if we step up together. Wear a mask. Stay socially distanced. Avoid large indoor gatherings. Each of us has a duty to do what we can to protect ourselves, our families, and our fellow Americans."]

                    myout.write("10.")

                    my_tweet = str(tweet10).split(".")

                    for i in range(0, len(my_tweet) - 1):
                        new_sentence = NewTweet(my_tweet[i], words_to_change[k], my_corpus)
                        myout.write(" < " + new_sentence.replace("['", "").replace('["', "") + " > \n")
                        k += 1

                    ##################################### part 3 #####################################################

                    songs_titles = []

                    artists_names = []

                    my_song_words = []

                    s = 0

                    for p in range(0, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song1_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song2_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song3_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song4_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song5_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song6_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song7_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song8_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song9_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song10_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song11_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    for p in range(s + 1, len(my_songs)):

                        if my_songs[p].startswith("=") and my_songs[p][1] == '=' and my_songs[p][2] == '=':

                            songs_titles.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p].startswith("=") and my_songs[p][1] == '=':

                            artists_names.append(my_songs[p].rstrip("\n").replace('=', ''))

                        elif my_songs[p] == '\n':
                            s = p
                            break

                        else:

                            my_song_words.append(my_songs[p].rstrip("\n"))

                    song12_words = ' '.join(my_song_words).split(' ')
                    my_song_words.clear()

                    ####### now make a list for each weight function #######

                    constant_list = []
                    randon_list = []
                    myWeights_list = []

                    # for i in range(1,12): now call every function with the right song words and append to the proper list:

                    ### 1 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song1_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song1_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song1_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 2 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song2_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song2_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song2_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 3 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song3_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song3_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song3_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 4 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song4_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song4_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song4_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 5 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song5_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song5_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song5_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 6 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song6_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song6_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song6_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 7 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song7_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song7_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song7_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 8 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song8_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song8_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song8_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 9 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song9_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song9_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song9_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 10 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song10_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song10_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song10_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 11 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song11_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song11_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song11_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ### 12 ####

                    const_avg = ConstantFunction(pre_trained_model_50, song12_words, 50)

                    rand_avg = RandomFunction(pre_trained_model_50, song12_words, 50)

                    myWeig_avg = MyWeightsFunction(pre_trained_model_50, song12_words, 50)

                    constant_list.append(const_avg)
                    randon_list.append(rand_avg)
                    myWeights_list.append(myWeig_avg)

                    ####### now use PCA ##############

                    ##### constant list #####

                    my_scaler = StandardScaler()

                    my_scaled_data = my_scaler.fit_transform(constant_list)

                    my_pca = PCA(n_components=2)

                    constant_output = my_pca.fit_transform(my_scaled_data)

                    ##### random list ######

                    my_scaled_data = my_scaler.fit_transform(randon_list)

                    random_output = my_pca.fit_transform(my_scaled_data)

                    ##### myWeights list #####

                    my_scaled_data = my_scaler.fit_transform(myWeights_list)

                    my_weights_output = my_pca.fit_transform(my_scaled_data)

                    #################################################

                    my_constant_graph = plt.figure()

                    my_constant_ax = my_constant_graph.add_subplot(111)

                    for i in range(0, 12):
                        my_x, my_y = constant_output[i][0], constant_output[i][1]

                        my_constant_ax.plot(my_x, my_y, 'bo')

                        random_num = np.random.uniform(1, 5)

                        my_constant_ax.text(my_x, my_y, artists_names[i] + '-' + songs_titles[i], fontsize=9)

                    my_constant_ax.set_title("Moataz Odeh - Constant weight")

                    ####################################################

                    my_random_graph = plt.figure()

                    my_random_ax = my_random_graph.add_subplot(111)

                    for i in range(0, 12):
                        my_x, my_y = random_output[i][0], random_output[i][1]

                        my_random_ax.plot(my_x, my_y, 'bo')

                        random_num = np.random.uniform(1, 5)

                        my_random_ax.text(my_x, my_y, artists_names[i] + '-' + songs_titles[i], fontsize=9)

                    my_random_ax.set_title("Moataz Odeh - Random weight")

                    ######################################################

                    my_weights_graph = plt.figure()

                    my_weights_ax = my_weights_graph.add_subplot(111)

                    for i in range(0, 12):
                        my_x, my_y = my_weights_output[i][0], my_weights_output[i][1]

                        my_weights_ax.plot(my_x, my_y, 'bo')

                        # random_num = np.random.uniform(1,5)

                        my_weights_ax.text(my_x, my_y, artists_names[i] + '-' + songs_titles[i], fontsize=9)

                    my_weights_ax.set_title("Moataz Odeh - My weight")

                    plt.show()
                    myout.write('\n - * - * - * - \n')
