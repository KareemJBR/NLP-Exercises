from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv


# The Corpus class from previous exercises:

class Corpus:

    def __init__(self):
        return


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
