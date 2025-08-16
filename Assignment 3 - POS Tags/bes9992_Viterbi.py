import math
from bes9992_HMM import hmm

# HANDLES OOV WORDS
def handle_oov(word):
    return 1e-7

# BUILD AND FILL VITERBI TABLE
def build_and_fill_viterbi_table(words, tag_counts, transition_probs, emission_probs, word_set):
    viterbi_table = []
    backpointer = []

    # Initialize first column (for the first word)
    first_col = {}
    bp_first_col = {}
    total_tags = sum(tag_counts.values())
    first_word = words[0]

    for tag in tag_counts:
        emission_prob = emission_probs[tag].get(first_word, handle_oov(first_word))
        prior_prob_of_tag = math.log(tag_counts[tag] / total_tags)
        first_col[tag] = prior_prob_of_tag + math.log(emission_prob)
        bp_first_col[tag] = None

    viterbi_table.append(first_col)
    backpointer.append(bp_first_col)

    # Fill in the rest of the Viterbi table for the remaining words
    for i in range(1, len(words)):
        viterbi_col = {}
        bp_col = {}
        current_word = words[i]

        for tag in tag_counts:
            max_prob = float('-inf')
            best_prev_tag = None

            for prev_tag in tag_counts:
                transition_prob = math.log(transition_probs.get(prev_tag, {}).get(tag, handle_oov(prev_tag)))
                emission_prob = math.log(emission_probs[tag].get(current_word, handle_oov(current_word)))
                prob = viterbi_table[i-1][prev_tag] + transition_prob + emission_prob

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag

            viterbi_col[tag] = max_prob
            bp_col[tag] = best_prev_tag

        viterbi_table.append(viterbi_col)
        backpointer.append(bp_col)

    return viterbi_table, backpointer

# FIND THE BEST PATH (sequence of POS tags)
def backtrack_best_path(viterbi_table, backpointer):
    best_path = []
    last_col = viterbi_table[-1]
    best_last_tag = max(last_col, key=last_col.get)
    best_path.append(best_last_tag)

    for i in range(len(viterbi_table) - 1, 0, -1):
        best_last_tag = backpointer[i][best_last_tag]
        best_path.insert(0, best_last_tag)

    return best_path

# MAIN VITERBI FUNCTION
def viterbi(words, transition_probs, emission_probs, tag_counts, word_set):
    viterbi_table, backpointer = build_and_fill_viterbi_table(words, tag_counts, transition_probs, emission_probs, word_set)
    best_path = backtrack_best_path(viterbi_table, backpointer)
    return best_path

if __name__ == "__main__":
    # TRAINING_HMM DATA --> TRANSITIONS AND EMISSION PROBABILITIES
    transition_probs, emission_probs, tag_counts, word_set = hmm('WSJ_02-21.pos')
    
    # INPUT & OUTPUT FILES
    training_file = 'WSJ_23.words'
    output_file = 'submission.pos'
    
    # READING TRAINING FILE
    with open(training_file, 'r') as f:
        sentences = f.read().strip().split('\n\n')
        
    # WRITE TO OUTPUT SUBMISSION FILE
    with open(output_file, 'w') as out_f:
        for sentence in sentences:
            words = sentence.split()
            tags = viterbi(words, transition_probs, emission_probs, tag_counts, word_set)
            for word, tag in zip(words, tags):
                out_f.write(f"{word}\t{tag}\n")  # making sure to put one tab in between word and tag
            out_f.write("\n")
