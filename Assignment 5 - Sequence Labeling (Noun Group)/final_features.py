# 1 corresponding line of features for each line in the input file 
# blank lines in input should correspond to blank lines in feature file
# each line should contain tab seperated values
    # 1st: token
    # 2nd: as many features as you want
    # 3rd: 
        # training file: BIO tag
        # test file: no final BIO field 

def pos_tags_after_last_dt(sentence, index):
    POS_tags = set()
    for word_POS in sentence[:index]:
        _, POS = word_POS.split()
        if POS == 'DT':
            POS_tags = set()
        else:
            POS_tags.add(POS)
    return '+'.join(sorted(POS_tags))


def extract_word_features(word, POS, sentence, index, training, prev_BIO, prev_POS, lines):
    # Curr word features
    features = [
        word,
        f"POS={POS}",
        f"has_capital={'true' if word[0].isupper() else 'false'}",
        f"word_pattern={'all_caps' if word.isupper() else 'no_caps' if word.islower() else 'title' if word.istitle() else 'mixed'}",
        f"has_number{'true' if any(char.isdigit() for char in word) else 'false'}",
        f"has_hyphen={'true' if '-' in word else 'false'}",
        f"word_length={len(word)}",
        f"first_letter={word[:1]}",
        f"first_two_letters={word[:2]}",
        f"last_letter={word[-1:]}",
        f"last_two_letters={word[-2:]}"
    ]

    # Prev word features
    if index > 0 and lines[index - 1].strip():
        prev_parts = lines[index - 1].strip().split()
        prev_word = prev_parts[0]
        prev_POS = prev_parts[1]
        features.append(f"PREV_WORD={prev_word}")
        features.append(f"PREV_POS={prev_POS}")
        features.append(f"PREV_POS+POS={prev_POS}+{POS}")
        if training:
            features.append(f"PREV_BIO={prev_BIO}")
    else:
        features.append("PREV_WORD=BOS")
        features.append(f"PREV_POS+POS=BOS+{POS}")

    # Next word features
    if index < len(lines) - 1 and lines[index + 1].strip():
        next_parts = lines[index + 1].strip().split()
        next_word = next_parts[0]
        next_POS = next_parts[1]
        features.append(f"NEXT_WORD={next_word}")
        features.append(f"NEXT_POS={next_POS}")
        features.append(f"POS+NEXT_POS={POS}+{next_POS}")
    else:
        features.append("NEXT_WORD=EOS")
        features.append(f"POS+NEXT_POS={POS}+EOS")

    # Tags since last determiner
    POS_tags_after_last_determiner = pos_tags_after_last_dt(sentence, len(sentence))
    features.append(f"POS_tags_after_dt={POS_tags_after_last_determiner}")

    # Prev 2 words features
    if index > 1 and lines[index - 2].strip():
        prev_prev_parts = lines[index - 2].strip().split()
        features.append(f"PREV2_WORD={prev_prev_parts[0]}")
        features.append(f"PREV2_POS={prev_prev_parts[1]}")
    else:
        features.append("PREV2_WORD=BOS")

    # Next 2 words features
    if index < len(lines) - 2 and lines[index + 2].strip():
        next_next_parts = lines[index + 2].strip().split()
        features.append(f"NEXT2_WORD={next_next_parts[0]}")
        features.append(f"NEXT2_POS={next_next_parts[1]}")
    else:
        features.append("NEXT2_WORD=EOS")

    return features


def process_sentence(sentence, out_file, training, lines):
    prev_POS = "BOS"
    prev_BIO = "O"

    for index, line in enumerate(lines):
        if line.strip() == "":
            out_file.write("\n")
            prev_POS = "BOS"
            prev_BIO = "O"
            sentence.clear()
            continue

        elements = line.strip().split()
        word = elements[0]
        POS = elements[1]
        if training:
            BIO_tag = elements[2]

        sentence.append(f"{word} {POS}")

        # Extract features for the curr word
        features = extract_word_features(word, POS, sentence, index, training, prev_BIO, prev_POS, lines)

        # Add BIO tag if training 
        if training:
            features.append(BIO_tag)
            # Update prev_BIO for next run of program 
            prev_BIO = BIO_tag

        # Write features to the output file
        out_file.write("\t".join(features) + "\n")


def extract_features(input_file, output_file, training=True):
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        lines = in_file.readlines()
        sentence = []
        process_sentence(sentence, out_file, training, lines)


def main():
    # Run program on training file
    extract_features("WSJ_02-21.pos-chunk", "training.feature", training=True)
    print("training.feature has been created.")

    # Run program on development file
    extract_features("WSJ_24.pos", "test.feature", training=False)
    print("test.feature has been created.")

    # Run program on test file (FOR SUBMISSION)
    # extract_features("WSJ_23.pos", "test.feature", training=False)
    # print("test.feature has been created.")


if __name__ == "__main__":
    main()
