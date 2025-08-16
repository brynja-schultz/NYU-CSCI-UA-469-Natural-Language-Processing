This TF-IDF based Information Retrieval System ranks documents by relevance to user queries using cosine similarity between TF-IDF vectors. 
It processes abstracts from cran.all.1400 and queries from cran.qry, returning the top 100 most relevant documents for each query.

To run the system, first ensure you have Python 3.x (I used Python 3.10.9) and NLTK installed (pip install nltk). 
The required files include cran.all.1400 (abstracts), cran.qry (queries), and stop_list.py (stop words). Then, run the following command:

python main.py --documents /path/to/cran.all.1400 --queries /path/to/cran.qry --output /path/to/output.txt

This will generate an output file where each line lists a query ID, document ID, and similarity score in the format <query_id> <document_id> <similarity_score>.

The system preprocesses the text by removing stop words and punctuation, calculates TF-IDF feature vectors, and ranks documents based on cosine similarity. 
Verify that NLTK’s tokenizer is downloaded (nltk.download('punkt')) before running this code.


WORK CITED:
https://youtu.be/vZAXpvHhQow?si=2tGQECYXpZJjPANQ
https://youtu.be/zcUGLp5vwaQ?si=An3Iu-H4umZgbbeT
https://youtu.be/k1tD7pYKWuM?si=4nOBpqbMqj9Ve3nu
https://www.geeksforgeeks.org/python-nltk-nltk-tokenizer-word_tokenize/
https://www.nltk.org/api/nltk.tokenize.html
https://www.geeksforgeeks.org/defaultdict-in-python/
