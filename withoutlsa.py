
import pandas as pd
import re
import nltk.classify.util
from nltk import pos_tag
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC


# x is your dataset
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

#raw_text_dataset = pd.read_csv("love_hate.csv")
#train_data = raw_text_dataset[0:1800]
#test_data = raw_text_dataset[1800:2586]


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
print(num_reviews)
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


print(clean_test_reviews)
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             ngram_range= (1,3),
                             max_features = 12000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.

train_data_features = vectorizer.fit_transform(postag_train_reviews)
print(train_data_features.shape)


test_data_features = vectorizer.transform(postag_test_reviews)
#test_data_features = test_data_features.toarray()
print(test_data_features.shape)

ch2 = SelectKBest(chi2, k=3000)
X_train = ch2.fit_transform(train_data_features, y_train_labels)
X_test = ch2.transform(test_data_features)

print("\nUsing Multinomial NB on SelectKBest features")
mnbb = MultinomialNB()
mnbb.fit(X_train,  y_train_labels)
result = mnbb.predict(X_test)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.confusion_matrix(y_test_labels,result))
#print(metrics.classification_report(y_test_labels, result))

print("\nUsing KNN on SelectKBest features")
knn_tfidf = KNeighborsClassifier(n_neighbors=20, algorithm='brute', metric='cosine')
knn_tfidf.fit(X_train, y_train_labels)
# Classify the test vectors.
p = knn_tfidf.predict(X_test)
print(metrics.accuracy_score(y_test_labels, p))
#print(metrics.confusion_matrix(y_test_labels,p))
#print(metrics.classification_report(y_test_labels, p))

print("\nUsing LinearSVC on SelectKBest features")
lsvc = LinearSVC()
lsvc.fit(X_train,  y_train_labels)
result = lsvc.predict(X_test)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.confusion_matrix(y_test_labels,result))
#print(metrics.classification_report(y_test_labels, result))

print("\nUsing LIBSVC on SelectKBest features")
libsvc = SVC(kernel= 'linear')
libsvc.fit(X_train,  y_train_labels)
result = libsvc.predict(X_test)
print(metrics.accuracy_score(y_test_labels, result))
#print(metrics.confusion_matrix(y_test_labels,result))
#print(metrics.classification_report(y_test_labels, result))
