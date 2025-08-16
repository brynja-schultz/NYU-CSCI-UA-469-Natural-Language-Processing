Features: 
    Current Word:  
        The word itself
        POS
        Whether there is a capital letter
        Pattern of the word
        Whether there is a hyphen
        Length of the word
        First letter of word
        First two letters of word
        Last letter of word
        Last two letters of word
        BIO (if training)
    Previous Word:
        The word itself
        POS
        BIO (if training)
    Next Word:
        The word itself
        POS
    Previous 2 Word:
        The word itself
        POS
    Next 2 Word:
        The word itself
        POS
    POS tags after the last determiner

Score on Development Corpus:
    Accuracy: 96.83
    Precision: 90.40
    Recall: 92.89
    F1: 9.16
    rounded to: 9

To run:
    Run python file to create the features files
        python final_features.py
    Compile the Java code
        javac -cp maxent-3.0.0.jar:trove.jar *.java
    Create model
        java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk
    Create system output
        java -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk
    Test / Score sytem output (only when using the development file)
        python score.chunk.py WSJ_24.pos-chunk response.chunk

Works Cited:
    https://www.nltk.org/book/ch07.html#ref-chunk-fe-paired 
