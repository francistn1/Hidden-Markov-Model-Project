############################################################
# CMPSC 442: Homework 6
############################################################

student_name = "Timothy Nicholl"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import collections
from collections import defaultdict


############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    with open(path) as file:
        return [[tuple(i.split('=')) for i in line.split()] for line in file]


class Tagger(object):

    def __init__(self, sentences):
        # It is imperative that you use Laplace smoothing where appropriate to ensure that your system
        # can handle novel inputs
        smooth_vector = 1e-10

        pi = {}
        pi_tagged = {}
        a = {}
        a_tagged = {}
        b = {}
        b_tagged = {}

        length = len(sentences)
        v_itag = len(pi_tagged)

        for sentence in sentences:
            # The initial tag probabilities π(ti) for 1 ≤ i ≤ n, where π(ti) is the probability that a sentence
            # begins with tag ti. We need to count the ones that are tagged
            if sentence[0][1] in pi_tagged:
                pi_tagged[sentence[0][1]] += 1
            else:
                pi_tagged[sentence[0][1]] = 1

            # The transition probabilities a(ti → tj) for 1 ≤ i, j ≤ n, where a(ti → tj ) is the probability that
            # tag tj occurs after tag ti.
            for i in range(len(sentence)):
                if i != len(sentence) - 1:
                    if sentence[i][1] in a_tagged:
                        if sentence[i + 1][1] in a_tagged[sentence[i][1]]:
                            a_tagged[sentence[i][1]][sentence[i + 1][1]] += 1
                        else:
                            a_tagged[sentence[i][1]][sentence[i + 1][1]] = 1
                    else:
                        a_tagged[sentence[i][1]] = {}
                        a_tagged[sentence[i][1]][sentence[i + 1][1]] = 1

                # The emission probabilities b(ti → wj) for 1 ≤ i ≤ n and 1 ≤ j ≤ m, where b(ti → wj) is the
                # probability that token wj is generated given tag ti.
                if sentence[i][1] in b_tagged:
                    if sentence[i][0] in b_tagged[sentence[i][1]]:
                        b_tagged[sentence[i][1]][sentence[i][0]] += 1
                    else:
                        b_tagged[sentence[i][1]][sentence[i][0]] = 1
                else:
                    b_tagged[sentence[i][1]] = {}
                    b_tagged[sentence[i][1]][sentence[i][0]] = 1

        for tag in b_tagged:
            b[tag] = {}
            word_dict = b_tagged[tag]
            v_word = len(word_dict)
            sum_word = sum(word_dict.values())
            for word in word_dict:
                word_count = word_dict[word]
                pw = (word_count + smooth_vector) / (sum_word + smooth_vector * (v_word + 1))
                b[tag][word] = math.log(pw)
            b[tag]["<UNK>"] = math.log(smooth_vector / (sum_word + smooth_vector * (v_word + 1)))

        for tag in a_tagged:
            a[tag] = {}
            post_tags = a_tagged[tag]
            v_post_dict = len(post_tags)
            sum_post_tags = sum(post_tags.values())
            for post_tag in post_tags:
                post_tag_count = post_tags[post_tag]
                p_post_tag = (post_tag_count + smooth_vector) / (sum_post_tags + smooth_vector * (v_post_dict + 1))
                a[tag][post_tag] = math.log(p_post_tag)
            a[tag]["<UNK>"] = math.log(smooth_vector / (sum_post_tags + smooth_vector * (v_post_dict + 1)))

        for init_tag in pi_tagged:
            total_tagged = pi_tagged[init_tag]
            pi_tag = (total_tagged + smooth_vector) / (length + smooth_vector * (v_itag + 1))
            pi[init_tag] = math.log(pi_tag)
            pi["<UNK>"] = math.log(smooth_vector / (length + smooth_vector * (v_itag + 1)))

            self.pi = pi
            self.a = a
            self.b = b
            self.states = b.keys()

    def most_probable_tags(self, tokens):
        result = []
        for token in tokens:
            max_val = -9999
            max_tag = "<START>"
            for t in self.b:
                if token in self.b[t]:
                    val = self.b[t][token]
                else:
                    val = self.b[t]["<UNK>"]

                if val > max_val:
                    max_val = val
                    max_tag = t

            result.append(max_tag)
        return result

    def viterbi_tags(self, tokens):
        v_tags = [{}]
        # initialization
        for curr_state in self.states:
            if tokens[0] in self.b[curr_state]:
                v_tags[0][curr_state] = self.pi[curr_state] + self.b[curr_state][tokens[0]]
            else:
                v_tags[0][curr_state] = self.pi[curr_state] + self.b[curr_state]["<UNK>"]
        # compute the probability of the most likely tag sequence
        for t in range(1, len(tokens)):
            v_tags.append({})
            for y in self.states:
                if tokens[t] in self.b[y]:
                    prob = max(v_tags[t-1][z] + self.a[z][y] + self.b[y][tokens[t]] for z in self.states)
                else:
                    prob = max(v_tags[t-1][z] + self.a[z][y] + self.b[y]["<UNK>"] for z in self.states)
                v_tags[t][y] = prob
            # reconstruct the sequence which achieves that probability from end to
            # beginning by tracing backpointers
        reconstructed_path = []
        for j in v_tags:
            for x,y in j.items():
                if j[x] == max(j.values()):
                    reconstructed_path.append(x)
        return reconstructed_path





############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
Approximately how long did you spend on this assignment?

6 hours, started back in late november
"""

feedback_question_2 = """
Which aspects of this assignment did you find most challenging? Were there any
significant stumbling blocks?

Understanding the logic was the most challenging aspect of this project and implementing it properly. Especially, the
init function, if you notice I have less comments than I did before thats because I worked quickly on this due to me 
running out of time for this assignment. But I managed to pull through and complete the assignment
"""

feedback_question_3 = """
Which aspects of this assignment did you like? Is there anything you would have
changed?

I liked my implementation of viterbi's algorithm, it was pretty straight forward after looking at the slides and 
reading tutorials online, one very helpful tutorial was this one I found here:
https://www.pythonpool.com/viterbi-algorithm-python/

"""
