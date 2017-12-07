from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Just like the previous example, here also we are going to deal
# 20 newsgroup data.
categories = ['alt.atheism',
              'talk.religion.misc',
              'comp.graphics',
              'sci.space']

# Loading training data
data_train = fetch_20newsgroups(subset='train',
                                categories=categories,
                                shuffle=True,
                                random_state=42)

# Loading test data
data_test = fetch_20newsgroups(subset='test',
                               categories=categories,
                               shuffle=True,
                               random_state=42)
y_train, y_test = data_train.target, data_test.target
feature_extractor_type = "tfidf"
if feature_extractor_type == "count":
    # The other vectorizer we can use is CountVectorizer
    # But for CountVectorizer we need to fit transform over
    # both training and test data as it requires the complete
    # vocabulary to create the matrix
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(data_train.data + data_test.data)
    X_train = vectorizer.transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
elif feature_extractor_type == "tfidf":
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

# alpha is additive (Laplace/Lidstone) smoothing parameter (0 for
# no smoothing).
clf = MultinomialNB(alpha=.01)

# Training the classifier
clf.fit(X_train, y_train)

# Predicting results
y_predicted = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_predicted)
print("accuracy: %0.3f" % score)
