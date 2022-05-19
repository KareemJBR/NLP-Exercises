from sys import argv
import numpy as np


class Rule:

    def __init__(self, source_non_terminal, destination, probability, destination_is_terminal):
        self.source = source_non_terminal
        self.dest = destination
        self.probability = probability
        self.destination_is_terminal = destination_is_terminal


class Parser:

    def __init__(self):
        self.rules = []

    def get_tree(self, sen):
        """Returns the parser tree with the highest probability for the received sentence as an argument, and the log of
         the probability."""
        sen_list = sen.split()

        probs_chart = np.zeros(shape=(len(sen_list), len(sen_list)))
        rules_indices_chart = np.zeros(shape=probs_chart.shape)
        sequences_chart = []

        pass


if __name__ == '__main__':

    input_grammar = argv[1]         # The name of the file that contains the probabilistic grammar
    input_sentences = argv[2]       # The name of the file that contains the input sentences (tests)
    output_trees = argv[3]          # The name of the output file

    output_text = ""

    parser = Parser()

    with open(input_grammar, 'r') as jabber:
        rules = jabber.readlines()

        for rule in rules:
            temp = rule.split()
            prob = temp[0]
            source = temp[1]

            is_terminal = (True if temp[3].islower() else False)

            dest = ""
            for i in range(3, len(temp) - 1):
                dest += temp[i] + " "
            dest += temp[-1]

            parser.rules.append(Rule(source, dest, prob, is_terminal))

    sentences = []

    with open(input_sentences, 'r') as jabber:
        lines = jabber.readlines()
        for line in lines:
            if len(line) == 0:
                continue

            if line[-1] == '\n':
                sentences.append(line[0:len(line) - 1])
            else:
                sentences.append(line)

    for sentence in sentences:
        output_text += 'Sentence: ' + sentence + '\n'
        output_text += parser.get_tree(sentence)
        output_text += '\n'

    with open(output_trees, 'w', encoding='utf-8') as jabber:
        jabber.write(output_text)
