This system implements a Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging using the Viterbi algorithm. 
The HMM is trained on a corpus of POS-tagged sentences, where each word in the training file is paired with its corresponding POS tag. 
From this training data, the system calculates two key probabilities: transition probabilities (the likelihood of one POS tag following another) 
and emission probabilities (the likelihood of a word being associated with a specific POS tag). These probabilities are used to predict POS tags for new, 
unseen sentences.

The Viterbi algorithm is applied to these unseen sentences, where it computes the most likely sequence of POS tags based on the observed words and the learned probabilities. 
The algorithm works by building a table to store the probability of each possible sequence of tags for the sentence, and a backpointer table to trace the best sequence of tags 
once the probabilities have been computed. The system handles out-of-vocabulary (OOV) words—words that do not appear in the training corpus—by assigning them a small default 
probability of `1e-7`. This ensures that unknown words do not disrupt the probability calculations, allowing the algorithm to still make reasonable predictions.

To run the system, first train the HMM using the `bes9992_HMM.py` script, which reads in a POS-tagged training file and outputs the transition and emission probabilities, 
as well as tag counts and a set of all words in the training data. Once the model is trained, the `bes9992_Viterbi.py` script can be run to predict POS tags for new sentences. 
This script reads an input file of untagged sentences, applies the Viterbi algorithm using the trained probabilities, and outputs a file where each word is paired with its predicted POS tag.

The system is designed to output results in a format where each word is followed by its corresponding predicted tag, with one word-tag pair per line, and sentences separated by blank lines. 
To ensure the generated output file is correct, the system checks the number of lines in the output file against a reference test file and raises an error if they do not match. 
This ensures that the output is correctly formatted and ready for evaluation.
