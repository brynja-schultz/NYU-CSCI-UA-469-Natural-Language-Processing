import math

def read_training_file(file):
    # USING DICTIONARIES TO STORE KEY AND VALUE PAIRS
    trans_probability = {} # KEY = PREV POS TAG, VALUE = (CURR POS TAG :  PROB OF TRANSITIONING FROM PREV TAG TO CURR TAG)
    emission_probs = {} # KEY = POS TAG, VALUE = (WORD : PROB OF WORD GIVEN THE TAG)
    num_tag = {} # KEY = POS TAG, VALUE = TOTAL COUNT OF POS TAG APPEARING 
    tag_bigrams = {} # KEY = PREV POS, VALUE: COUNT OF HOW MANY BIGRAMS START WITH THIS TAG

    # SET TO STORE ALL DISTINCT WORDS (WORD & POS)
    word_set = set()

    prev_tag = None

    # READING TRAINING FILE
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()

            # IGNORE BLANK LINES
            if line == "":
                prev_tag = None
                continue

            word, tag = line.split()
            word_set.add(word)

            # CALL THE FUNCTIONS
            update_tag_counts(tag, num_tag)
            update_emission_probs(word, tag, emission_probs)
            update_trans_probability(prev_tag, tag, trans_probability, tag_bigrams)
            
            # UPDATE PREV TAG FOR NEXT LINE
            prev_tag = tag

    return trans_probability, emission_probs, num_tag, tag_bigrams, word_set

def update_tag_counts(tag, num_tag):
    # UPDATE TAG COUNTS
    if tag in num_tag:
        num_tag[tag] += 1
    else:
        num_tag[tag] = 1

# UPDATE EMISSION PROBABILITIES 
def update_emission_probs(word, tag, emission_prob):
    # IF POS TAG DOES NOT EXIST --> CREATE ENTRY 
    if tag not in emission_prob:
        emission_prob[tag] = {}
    # IF WORD IS ALREADY STORED --> UPDATE COUNT
    if word in emission_prob[tag]:
        emission_prob[tag][word] += 1
    # IF WORD IS NEW --> CREATE COUNT 
    else:
        emission_prob[tag][word] = 1

def update_trans_probability(prev_tag, curr_tag, trans_probability, tag_bigrams):
    # UPDATE TRANSITION PROBABILITIES
    if prev_tag is not None:
        # IF PREV TAG DOES NOT EXIST --> CREATE ENTRY
        if prev_tag not in trans_probability:
                trans_probability[prev_tag] = {}
        # IF PREV TAG EXISTS --> UPDATE COUNT
        if curr_tag in trans_probability[prev_tag]:
            trans_probability[prev_tag][curr_tag] += 1
        # IF PREV TAG DOES NOT EXIST --> CREATE COUNT
        else:
            trans_probability[prev_tag][curr_tag] = 1
                
        # PREV TAG EXISTS --> UPDATE COUNT
        if prev_tag in tag_bigrams:
            tag_bigrams[prev_tag] += 1
        # PREV TAG DOES NOT EXIST --> CREATE COUNT 
        else:
            tag_bigrams[prev_tag] = 1

def convert_counts_to_probabilities(trans_probability, emission_probs, num_tag, tag_bigrams):
    for tag, next_tags in trans_probability.items():
        # COVERT RAW COUNTS TO TRANSITION PROBABILITIES 
        for t, count in next_tags.items():
            trans_probability[tag][t] = count / tag_bigrams[tag]
    
    # CONVERT RAW COUNTS TO EMISSION PROBABILITIES 
    for tag, words in emission_probs.items():
        for word, count in words.items():
            emission_probs[tag][word] = count / num_tag[tag]

    return trans_probability, emission_probs

def hmm(file):
    trans_probability, emission_probs, num_tag, tag_bigrams, word_set = read_training_file(file)
    trans_probability, emission_probs = convert_counts_to_probabilities(trans_probability, emission_probs, num_tag, tag_bigrams)
    return trans_probability, emission_probs, num_tag, word_set

if __name__ == "__main__":
    file = 'WSJ_02-21.pos'
    # GET VALUES TO USE FOR VITERBI PROGRAM
    trans_probability, emission_probs, num_tag, word_set = hmm(file)
