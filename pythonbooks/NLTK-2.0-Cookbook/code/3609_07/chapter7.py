'''
=================================
Training a Naive Bayes Classifier
=================================

>>> from nltk.corpus import movie_reviews
>>> from featx import label_feats_from_corpus, split_label_feats
>>> movie_reviews.categories()
['neg', 'pos']
>>> lfeats = label_feats_from_corpus(movie_reviews)
>>> lfeats.keys()
['neg', 'pos']
>>> train_feats, test_feats = split_label_feats(lfeats)
>>> len(train_feats)
1500
>>> len(test_feats)
500

>>> from nltk.classify import NaiveBayesClassifier
>>> nb_classifier = NaiveBayesClassifier.train(train_feats)
>>> nb_classifier.labels()
['neg', 'pos']

>>> negfeat = bag_of_words(['the', 'plot', 'was', 'ludicrous'])
>>> nb_classifier.classify(negfeat)
'neg'
>>> posfeat = bag_of_words(['kate', 'winslet', 'is', 'accessible'])
>>> nb_classifier.classify(posfeat)
'pos'

>>> from nltk.classify.util import accuracy
>>> accuracy(nb_classifier, test_feats)
0.72799999999999998

>>> probs = nb_classifier.prob_classify(test_feats[0][0])
>>> probs.samples()
['neg', 'pos']
>>> probs.max()
'pos'
>>> probs.prob('pos')
0.99999996464309127
>>> probs.prob('neg')
3.5356889692409258e-08

>>> nb_classifier.most_informative_features(n=5)
[('magnificent', True), ('outstanding', True), ('insulting', True), ('vulnerable', True), ('ludicrous', True)]

>>> from nltk.probability import LaplaceProbDist
>>> nb_classifier = NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)
>>> accuracy(nb_classifier, test_feats)
0.71599999999999997

>>> from nltk.probability import DictionaryProbDist
>>> label_probdist = DictionaryProbDist({'pos': 0.5, 'neg': 0.5})
>>> true_probdist = DictionaryProbDist({True: 1})
>>> feature_probdist = {('pos', 'yes'): true_probdist, ('neg', 'no'): true_probdist}
>>> classifier = NaiveBayesClassifier(label_probdist, feature_probdist)
>>> classifier.classify({'yes': True})
'pos'
>>> classifier.classify({'no': True})
'neg'


===================================
Training a Decision Tree Classifier
===================================

>>> from nltk.classify import DecisionTreeClassifier
>>> dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
>>> accuracy(dt_classifier, test_feats)
0.68799999999999994

>>> from nltk.probability import FreqDist, MLEProbDist, entropy
>>> fd = FreqDist({'pos': 30, 'neg': 10})
>>> entropy(MLEProbDist(fd))
0.81127812445913283
>>> fd['neg'] = 25
>>> entropy(MLEProbDist(fd))
0.99403021147695647
>>> fd['neg'] = 30
>>> entropy(MLEProbDist(fd))
1.0
>>> fd['neg'] = 1
>>> entropy(MLEProbDist(fd))
0.20559250818508304


=====================================
Training a Maximum Entropy Classifier
=====================================

>>> from nltk.classify import MaxentClassifier
>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='iis', trace=0, max_iter=1, min_lldelta=0.5)
>>> accuracy(me_classifier, test_feats)
0.5

>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='cg', trace=0, max_iter=10)
>>> accuracy(me_classifier, test_feats)
0.85599999999999998

>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
[Found megam: /usr/local/bin/megam]
>>> accuracy(me_classifier, test_feats)
0.86799999999999999


==============================================
Measuring Precision and Recall of a Classifier
==============================================

>>> from classification import precision_recall
>>> nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
>>> nb_precisions['pos']
0.6413612565445026
>>> nb_precisions['neg']
0.9576271186440678
>>> nb_recalls['pos']
0.97999999999999998
>>> nb_recalls['neg']
0.45200000000000001

>>> me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
>>> me_precisions['pos']
0.8801652892561983
>>> me_precisions['neg']
0.85658914728682167
>>> me_recalls['pos']
0.85199999999999998
>>> me_recalls['neg']
0.88400000000000001


==================================
Calculating High Information Words
==================================

>>> from featx import high_information_words, bag_of_words_in_set
>>> labels = movie_reviews.categories()
>>> labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
>>> high_info_words = set(high_information_words(labeled_words))
>>> feat_det = lambda words: bag_of_words_in_set(words, high_info_words)
>>> lfeats = label_feats_from_corpus(movie_reviews, feature_detector=feat_det)
>>> train_feats, test_feats = split_label_feats(lfeats)

>>> nb_classifier = NaiveBayesClassifier.train(train_feats)
>>> accuracy(nb_classifier, test_feats)
0.91000000000000003
>>> nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
>>> nb_precisions['pos']
0.89883268482490275
>>> nb_precisions['neg']
0.92181069958847739
>>> nb_recalls['pos']
0.92400000000000004
>>> nb_recalls['neg']
0.89600000000000002

>>> me_classifier = MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
>>> accuracy(me_classifier, test_feats)
0.88200000000000001
>>> me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
>>> me_precisions['pos']
0.88663967611336036
>>> me_precisions['neg']
0.87747035573122534
>>> me_recalls['pos']
0.876
>>> me_recalls['neg']
0.88800000000000001

>>> dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, depth_cutoff=20, support_cutoff=20, entropy_cutoff=0.01)
>>> accuracy(dt_classifier, test_feats)
0.68600000000000005
>>> dt_precisions, dt_recalls = precision_recall(dt_classifier, test_feats)
>>> dt_precisions['pos']
0.6741573033707865
>>> dt_precisions['neg']
0.69957081545064381
>>> dt_recalls['pos']
0.71999999999999997
>>> dt_recalls['neg']
0.65200000000000002


=================================
Combining Classifiers with Voting
=================================

>>> from classification import MaxVoteClassifier
>>> mv_classifier = MaxVoteClassifier(nb_classifier, dt_classifier, me_classifier)
>>> mv_classifier.labels()
['neg', 'pos']
>>> accuracy(mv_classifier, test_feats)
0.89600000000000002
>>> mv_precisions, mv_recalls = precision_recall(mv_classifier, test_feats)
>>> mv_precisions['pos']
0.8928571428571429
>>> mv_precisions['neg']
0.89919354838709675
>>> mv_recalls['pos']
0.90000000000000002
>>> mv_recalls['neg']
0.89200000000000002


============================================
Classifying with Multiple Binary Classifiers
============================================

>>> from nltk.corpus import reuters
>>> len(reuters.categories())
90

>>> from featx import reuters_high_info_words, reuters_train_test_feats
>>> rwords = reuters_high_info_words()
>>> featdet = lambda words: bag_of_words_in_set(words, rwords)
>>> multi_train_feats, multi_test_feats = reuters_train_test_feats(featdet)

>>> from classification import train_binary_classifiers
>>> trainf = lambda train_feats: MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
>>> labelset = set(reuters.categories())
>>> classifiers = train_binary_classifiers(trainf, multi_train_feats, labelset)
>>> len(classifiers)
90

>>> from classification import MultiBinaryClassifier, multi_metrics
>>> multi_classifier = MultiBinaryClassifier(*classifiers.items())

>>> multi_precisions, multi_recalls, avg_md = multi_metrics(multi_classifier, multi_test_feats)
>>> avg_md
0.18191264129488705

>>> multi_precisions['zinc']
1.0
>>> multi_recalls['zinc']
0.46153846153846156
>>> len(reuters.fileids(categories=['zinc']))
34

>>> multi_precisions['sunseed']
1.0
>>> multi_recalls['sunseed']
0.20000000000000001
>>> len(reuters.fileids(categories=['sunseed']))
16

>>> multi_precisions['rand']
>>> multi_recalls['rand']
0.0
>>> len(reuters.fileids(categories=['rand']))
3
'''

if __name__ == '__main__':
	import doctest
	doctest.testmod()