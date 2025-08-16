import argparse
import nltk
import string
import math

nltk.download('punkt')

from stop_list import closed_class_stop_words

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Function to split documents by their ID from abstracts or queries
def split_doc_by_id(content):
    return content.split('.I ')[1:]

# Function to extract sections from a document
def split_into_parts(lines):
    parts = {"title": "", "author": "", "bib_info": "", "abstract": ""}
    curr_part = None

    for line in lines:
        line = line.strip()
        if line.startswith(".T"):
            curr_part = "title"
        elif line.startswith(".A"):
            curr_part = "author"
        elif line.startswith(".B"):
            curr_part = "bib_info"
        elif line.startswith(".W"):
            curr_part = "abstract"
        else:
            if curr_part:
                parts[curr_part] += line + " "

    return parts

# Function to process each document (abstracts) 
def process_abstract(doc_content):
    lines = doc_content.strip().split("\n")
    doc_id = lines[0]
    parts = split_into_parts(lines[1:])
    return {
        "doc_id": doc_id,
        "title": parts.get("title", "").strip(),
        "author": parts.get("author", "").strip(),
        "bib_info": parts.get("bib_info", "").strip(),
        "abstract": parts.get("abstract", "").strip()
    }

# Process the abstracts from 'cran.all.1400'
def process_abstracts(file):
    content = read_file(file)
    docs = split_doc_by_id(content)
    return [process_abstract(doc) for doc in docs]

# Extract the actual query text
def extract_query_text(lines):
    text = ""
    curr_part = None

    for line in lines:
        line = line.strip()
        if line.startswith(".W"):
            curr_part = "abstract"
        elif curr_part == "abstract":
            text += line + " "

    return text

# Process each query
def process_query(query_content, mapped_id):
    lines = query_content.strip().split("\n")
    query_id = lines[0].strip()
    text = extract_query_text(lines[1:])
    return {
        "mapped_id": mapped_id,
        "query_id": query_id,
        "abstract": text.strip()
    }

# Process the queries from 'cran.qry'
def process_cran_queries(filename):
    content = read_file(filename)
    queries = split_doc_by_id(content)
    return [process_query(query, idx + 1) for idx, query in enumerate(queries)]

# Preprocess text (tokenize, remove stop words and punctuation)
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word not in closed_class_stop_words and
            word not in string.punctuation and not word.isdigit()]

# Update document frequencies for each term
def update_doc_frequencies(processed_doc, doc_frequencies):
    for term in set(processed_doc):
        if term not in doc_frequencies:
            doc_frequencies[term] = 0
        doc_frequencies[term] += 1

# Calculate document frequencies for all abstracts
def calc_doc_frequencies(documents):
    doc_frequencies = {}
    for doc in documents:
        update_doc_frequencies(preprocess_text(doc['abstract']), doc_frequencies)
    return doc_frequencies

# Calculate term frequencies for an abstract
def calc_term_frequencies(processed_abstract):
    term_frequencies = {}
    for word in processed_abstract:
        term_frequencies[word] = term_frequencies.get(word, 0) + 1
    return term_frequencies

# Calculate inverse document frequency (IDF)
def calc_idf(df_value, total_docs):
    return math.log((total_docs + 1) / (df_value + 1)) if df_value > 0 else 0

# Calculate TF-IDF feature vector
def calc_feature_vector(term_frequencies, doc_frequencies, total_docs):
    feature_vector = {}
    for word in term_frequencies.keys():
        term_frequencies_value = term_frequencies[word]
        idf = calc_idf(doc_frequencies.get(word, 0), total_docs)
        feature_vector[word] = term_frequencies_value * idf
    return feature_vector

# Calculate the TF-IDF vector for a given abstract/query
def calc_tf_idf(inp, df, total_docs):
    processed_abstract = preprocess_text(inp['abstract'])
    tf = calc_term_frequencies(processed_abstract)
    return calc_feature_vector(tf, df, total_docs)

# Calculate dot product for cosine similarity
def dot_product(query_vec, doc_vec):
    return sum(query_vec[word] * doc_vec.get(word, 0) for word in query_vec.keys())

# Calculate magnitude for cosine similarity
def magnitude(vector):
    return math.sqrt(sum(value ** 2 for value in vector.values()))

# Calculate cosine similarity between query and document vectors
def calc_cosine_similarity(query, doc):
    dot_product_value = dot_product(query['TF-IDF'], doc['TF-IDF'])
    query_magnitude = magnitude(query['TF-IDF'])
    doc_magnitude = magnitude(doc['TF-IDF'])

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    return dot_product_value / (query_magnitude * doc_magnitude)

# Argument parsing for file paths
def parse_arguments():
    parser = argparse.ArgumentParser(description="TF-IDF based Ad Hoc Information Retrieval System")
    parser.add_argument('--documents', type=str, required=True, help="Path to cran.all.1400 file (abstracts)")
    parser.add_argument('--queries', type=str, required=True, help="Path to cran.qry file (queries)")
    parser.add_argument('--output', type=str, required=True, help="Path to output file where results will be written")
    return parser.parse_args()

# Get top 100 results for a query
def get_results(query, documents):
    all_results = [(doc['doc_id'], calc_cosine_similarity(query, doc)) for doc in documents]
    all_results.sort(key=lambda tup: tup[1], reverse=True)
    top_results = all_results[:100]
    if len(top_results) < 100:
        top_results += [(doc['doc_id'], 0) for doc in documents[len(top_results):100]]
    return top_results

# Write the final results to the output file
def write_results(queries, documents, output_file):
    with open(output_file, 'w') as file:
        for query in queries:
            results = get_results(query, documents)
            for doc_id, similarity in results:
                file.write(f"{query['mapped_id']} {doc_id} {similarity}\n")

# Main function to process documents and queries and calculate similarity
def main():
    args = parse_arguments()
    documents = process_abstracts(args.documents)  # Abstracts (cran.all.1400)
    queries = process_cran_queries(args.queries)  # Queries (cran.qry)
    df = calc_doc_frequencies(documents)
    total_docs = len(documents)

    for query in queries:
        query['TF-IDF'] = calc_tf_idf(query, df, total_docs)

    for document in documents:
        document['TF-IDF'] = calc_tf_idf(document, df, total_docs)

    write_results(queries, documents, args.output)

if __name__ == '__main__':
    main()
