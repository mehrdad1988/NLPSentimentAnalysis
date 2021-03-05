import requests
from bs4 import BeautifulSoup
import re
from urllib.request import urlopen
from lxml import html

from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv

import seaborn as sns




def get_wordnet_pos(pos_tag):
    if pos_tag.startswith ('J'):
        return wordnet.ADJ
    elif pos_tag.startswith ('V'):
        return wordnet.VERB
    elif pos_tag.startswith ('N'):
        return wordnet.NOUN
    elif pos_tag.startswith ('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # lower text
    text = text.lower ()
    # tokenize text and remove puncutation
    text = [word.strip (string.punctuation) for word in text.split (" ")]
    # remove words that contain numbers
    text = [word for word in text if not any (c.isdigit () for c in word)]
    # remove stop words
    stop = stopwords.words ('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len (t) > 0]
    # pos tag text
    pos_tags = pos_tag (text)
    # lemmatize text
    text = [WordNetLemmatizer ().lemmatize (t[0], get_wordnet_pos (t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len (t) > 1]
    # join all
    text = " ".join (text)
    return (text)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40,
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()

URL = 'https://www.cancerresearchuk.org/about-cancer/cancer-chat/thread/blood/mucus-stool?items_per_page=50'
headers = {'Content-Type': 'text/html',}
page = requests.get(URL,headers=headers)
tree = html.fromstring(page.content)



myres=[]
reviews = []

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.findAll('div',class_='field-items')
#print(results)
for res in results:
   comments=res.findAll('div', class_='field-item even')
   for com in comments:
       comment = com.find('p')
       #print(comment)
       comment_str=str(comment)

       temp=re.compile (r'<[^>]+>').sub ('', comment_str)
       if(temp != "None"):
        myres.append(clean_text(temp))
        print(temp)

    #clean_output = clean_text (i)
    #reviews["review_clean"].append(i)
    #reviews["review"].append (i)

with open ('reviews.csv', mode='w' , encoding='utf-8' ) as csv_file:
    fieldnames = ['review', 'is_bad_review']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames )
    writer.writeheader()
    for i in myres:
        writer.writerow({'review': i , 'is_bad_review': "0"})

reviews = pd.read_csv("reviews.csv")
reviews = reviews.sample(frac = 0.1, replace = False, random_state=42)

reviews["review_clean"] = reviews["review"].apply(lambda x: clean_text(x))


sid = SentimentIntensityAnalyzer()
reviews["sentiments"] = reviews["review"].apply(lambda x: sid.polarity_scores(x))
reviews = pd.concat([reviews.drop(['sentiments'], axis=1), reviews['sentiments'].apply(pd.Series)], axis=1)

reviews["nb_chars"] = reviews["review"].apply(lambda x: len(x))
reviews["nb_words"] = reviews["review"].apply(lambda x: len(x.split(" ")))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews["review_clean"].apply(lambda x: x.split(" ")))]

model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

doc2vec_df = reviews["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews = pd.concat([reviews, doc2vec_df], axis=1)

tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews.index
reviews = pd.concat([reviews, tfidf_df], axis=1)

# print wordcloud
show_wordcloud(reviews["review"])

# Print Positive reviews - five top comments
reviews[reviews["nb_words"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)

# Print Negative reviews - five top comments
reviews[reviews["nb_words"] >= 5].sort_values("neg", ascending = False)[["review", "neg"]].head(10)

# plot sentiment distribution for positive and negative reviews

for x in [0, 1]:
    subset = reviews[reviews['is_bad_review'] == x]

    # Draw the density plot
    if x == 0:
        label = "Good reviews"
    else:
        label = "Bad reviews"
    sns.distplot (subset['compound'], hist=False, label=label)