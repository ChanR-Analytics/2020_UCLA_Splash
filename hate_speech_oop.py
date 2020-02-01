import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.stem.porter import *
from os import getcwd, listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from scikitplot.metrics import plot_confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
warnings.simplefilter(action='ignore', category=FutureWarning)


stopwords = nltk.corpus.stopwords.words('english')

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

sentiment_analyzer = VS()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features_(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.
    This is modified to only include those features in the final
    model."""

    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet) #Get text only

    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['compound'],
                twitter_objs[2], twitter_objs[1],]
    #features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    for t in tweets:
        feats.append(other_features_(t))
    return np.array(feats)


class HatebaseTwitter():
    def __init__(self, data_path):
        # Reading the Data Frame from the Hatebase Twitter CSV File
        self.df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")

    def eda(self):
        # Seeing the Number of CrowdFlower Annotators that Decided On a Tweet's Label
        annot_count = self.df['count'].value_counts().to_dict()

        ## Visualizing the Annotator Count
        plt.figure(figsize=(12,12))
        plt.bar(list(annot_count.keys()), list(annot_count.values()))
        plt.xlabel('Number of CrowdFlower Annotators', size=14)
        plt.ylabel('Number of Tweets Annotated', size=14)
        plt.show()

        # Visualizing the Proportion of Tweet Classification Labels and Class Imbalance
        labels = ['Hateful', 'Offensive', 'Neither']
        class_vals = self.df['class'].value_counts().to_dict()
        class_vals = {labels[i]: class_vals[i] for i in class_vals.keys()}
        fig, ax = plt.subplots(figsize=(12,12))
        ax.pie(class_vals.values(), labels=class_vals.keys(), autopct='%1.1f%%')
        ax.axis('equal')
        plt.title("Proportion of Tweet Classes", size=14)
        plt.show()

    def features(self):
        tweets = self.df['tweet'].tolist()
        vectorizer = TfidfVectorizer(
            tokenizer = tokenize,
            preprocessor = preprocess,
            ngram_range = (1,3),
            stop_words = stopwords,
            use_idf = True,
            smooth_idf = False,
            norm = None,
            decode_error = 'replace',
            max_features = 10000,
            min_df = 5,
            max_df = 0.75
        )

        # Constructing the TF IDF Matrix and Getting the Relevant Scores
        tfidf = vectorizer.fit_transform(tweets).toarray()
        vocab = {v:i for i,v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i: idf_vals[i] for i in vocab.values()}

        # Getting POS Tags for Tweets and Saving as a String
        tweet_tags = []
        for t in tweets:
            tokens = basic_tokenize(preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)

        # Use TF IDF vectorizer to get a token matrix for the POS tags
        pos_vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=None,
            preprocessor=None,
            ngram_range = (1,3),
            stop_words=None,
            use_idf=False,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.75,
        )

        # Construct POS TF Matrix and Get Vocabulary Dictionary
        pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
        pos_vocab = {v:i for i,v in enumerate(pos_vectorizer.get_feature_names())}

        # Getting the Other Features
        other_feature_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                                "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                                "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

        feats = get_feature_array(tweets)

        # Join Features Together
        M = np.concatenate([tfidf, pos, feats], axis=1)

        # Getting a List of Variable Names
        variables = ['']*len(vocab)
        for k,v in vocab.items():
            variables[v] = k

        pos_variables = ['']*len(pos_vocab)
        for k,v in pos_vocab.items():
            pos_variables[v] = k

        self.feature_names = variables + pos_variables + other_feature_names

        return M

    def l1_dim_reduce(self, M):
        df = self.df
        y = df['class']
        X = pd.DataFrame(M)
        dim_reduce = SelectFromModel(LogisticRegression(solver='liblinear', class_weight='balanced', C=0.04, penalty='l1'))
        X_ = dim_reduce.fit_transform(X, y)
        return X_

    def rfe_dim_reduce(self, M, classifier, n):
        df = self.df
        y = df['class']
        X = pd.DataFrame(M)
        if classifier == 'lr':
            model = LogisticRegression()
        elif classifier == 'rf':
            model = RandomForestClassifier()
        elif classifier == 'ada':
            model = AdaBoostClassifier()
        elif classifier == 'xgb':
            model = XGBClassifier()
        elif classifier == 'svc':
            model = LinearSVC()
        elif classifier == 'davidson':
            model = LogisticRegression(solver='liblinear', class_weight='balanced', C=0.04, penalty='l1')
        else:
            raise TypeError("Please use a proper classifier.")
        dim_reduce = RFE(model, step=n)
        X_ = dim_reduce.fit_transform(X, y)
        return X_

    def classify(self, X, type: str, classifier: str, test_prop: float, res: None, res_method: None):

        if type == 'binary':
            y = self.df['class'].replace(0,1)
        elif type == 'multi':
            y = self.df['class']
        else:
            raise TypeError("Choose a proper type of classification")

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_prop, stratify=y)

        if res == True:
            if res_method == 'down':
                nm = NearMiss()
                X_res, Y_res = nm.fit_resample(X_train, Y_train)
            elif res_method == 'up':
                sm = ADASYN()
                X_res, Y_res = sm.fit_resample(X_train, Y_train)
            else:
                raise TypeError("Resampling method not provided. Please use 'up' for oversampling or 'down' for undersampling.")

        if classifier == 'lr':
            model = LogisticRegression(solver='liblinear', class_weight='balanced', C=0.04, penalty='l2')
        elif classifier == 'svc':
            model = LinearSVC(C=0.004, penalty='l2')
        elif classifier == 'rf':
            n_est = int(input("Type in number of trees to estimate from: ").strip())
            model = RandomForestClassifier(n_estimators=n_est, bootstrap=True, max_depth=5)
        elif classifier == 'xgb':
            n_est = int(input("Type in number of trees to estimate from: ").strip())
            model = XGBClassifier(n_estimators=n_est, bootstrap=True, max_depth=5, reg_lamba=0.4)
        elif classifier == 'ada':
            n_est = int(input("Type in number of trees to estimate from: ").strip())
            model = AdaBoostClassifier(n_estimators=n_est, learning_rate=0.005)
        else:
            raise TypeError("Choose a proper classifier. Possible inputs: 'lr', 'svc', 'rf', 'xgb', 'ada' .")

        if res == True:
            model.fit(X_res, Y_res)
        else:
            model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        # Accuracy Percentage
        print(f"Accuracy is {round(accuracy_score(Y_test, Y_pred), 2)*100}%")

        # Classification Report
        print(classification_report(Y_pred, Y_test))

        # Matthew's Correlation Coefficient
        print(f"Matthew's Correlation Coefficient is {matthews_corrcoef(Y_test, Y_pred)}")

        # Plots of Confusion Matrix and ROC Curve
        plot_confusion_matrix(Y_test, Y_pred, figsize=(10,10)) 

        return model
