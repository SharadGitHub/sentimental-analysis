
import time
import pandas as pd
import re
import nltk.classify.util
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from nltk import pos_tag
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, chi2

###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

#raw_text_dataset = pd.read_csv("love_hate.csv")

#train_data = raw_text_dataset[0:1800]
#test_data = raw_text_dataset[1800:2586]

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

print(len(train_data))
print(len(test_data))

X_train_raw = train_data['content']
y_train_labels = train_data['sentiment']
X_test_raw = test_data['content']
y_test_labels = test_data['sentiment']


def review_to_words( review_text ):

    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = nltk.corpus.stopwords.words('english')
    stops += ['.', '://', 'http', 'com', '...']
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


print("Cleaning and parsing the training set tweets...\n")
num_reviews = X_train_raw.size
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print( "Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( review_to_words( X_train_raw[i] ))

postag_train_reviews = []
for d in clean_train_reviews:
        data_tokens = nltk.wordpunct_tokenize(d)
        temp = ["%s_%s" % (w,t) for w, t in pos_tag(data_tokens)]
        postag_train_reviews.append(" ".join(temp))


print("Cleaning and parsing the testing set tweets...\n")
test_tweets = X_test_raw.size
clean_test_reviews = []
for i in range( 0 , test_tweets ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print( "Review %d of %d\n" % ( i+1, test_tweets ))
    clean_test_reviews.append( review_to_words( X_test_raw[i] ))


postag_test_reviews = []
for d in clean_test_reviews:
        data_tokens = nltk.wordpunct_tokenize(d)
        temp = ["%s_%s" % (w,t) for w, t in pos_tag(data_tokens)]
        postag_test_reviews.append(" ".join(temp))




###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################

# Tfidf vectorizer:
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.
vectorizer = TfidfVectorizer( min_df=2, max_features=20000 ,stop_words=None, ngram_range= (1,3),
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it
# ("transform").
X_train_tfidf = vectorizer.fit_transform(postag_train_reviews)

print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first 150 principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(800)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)


print("  done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(postag_test_reviews)
X_test_lsa = lsa.transform(X_test_tfidf)

print(" train tfidf size",X_train_tfidf.size)
print("test tfidf size" ,X_test_tfidf.size)


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nUsing KNN on LSA vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
knn_lsa = KNeighborsClassifier(n_neighbors=20, algorithm='brute', metric='cosine')
knn_lsa.fit(X_train_lsa, y_train_labels)

# Classify the test vectors.
p = knn_lsa.predict(X_test_lsa)


# Measure accuracy

print(metrics.accuracy_score(y_test_labels, p))
#print(metrics.classification_report(y_test_labels, p))
# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("    done in %.3fsec" % elapsed)



print("\nUsing LinearSVC on LSA features")
lsvc = LinearSVC()
lsvc.fit(X_train_lsa,  y_train_labels)
result = lsvc.predict(X_test_lsa)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.confusion_matrix(y_test_labels,result))
#print(metrics.classification_report(y_test_labels, result))

print("\nUsing LIBSVC on LSA features")
libsvc = SVC(kernel= 'linear')
libsvc.fit(X_train_lsa,  y_train_labels)
result = libsvc.predict(X_test_lsa)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.classification_report(y_test_labels, result))

print("\nUsing MNB on tfidf")
mnb = MultinomialNB()
mnb.fit(X_train_tfidf,  y_train_labels)
result = mnb.predict(X_test_tfidf)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.classification_report(y_test_labels, result))
