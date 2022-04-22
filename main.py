# Designed & coded by Sandra Ashipala <https://github.com/sandramsc> 
import pandas as pd
import re
import string
import seaborn
import numpy as np
import nltk.data

pd.set_option('display.max_colwidth', None)
#handle missing values of type NaN, N/a, na
missing_value=["NaN", "N/a", "na", np.nan]
#import csv file
data = pd.read_csv('playmeditopia_us.csv', na_values=missing_value)
#data.isnull().sum()
#data.isnull().any()

#heatmap
#sns.heatmap(data.isnull(), yticklabels=False)

#drop NaN values that have all rows empty
#data_dropped = data.dropna()
data_dropped = data.dropna(how="all")

#remove URLs
def fix_URL(userImage):
	url = re.compile(r'https?://\S+|www\.\S+')
	return url.sub(r'', userImage)

data['userImage'] = data['userImage'].apply(fix_URL)
#data.head()

#remove emojies
def remove_emojis(data):
	emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
	return re.sub(emoj, '', data)
data['content'] = data['content'].apply(remove_emojis)

#remove punctuations
def remove_punct(content):
	data=str.maketrans('','',string.punctuation)
	return content.translate(data)

data['content'] = data['content'].apply(remove_punct)

#load lang specific file
tokenizer = nltk.data.load('.\Sandra\AppData\Roaming\nltk_data\tokenizers\punkt\PY3\english.pickle')
tokenizer = nltk.data.load('.\Sandra\AppData\Roaming\nltk_data\tokenizers\punkt\PY3\turkish.pickle')
tokenizer = nltk.data.load('.\Sandra\AppData\Roaming\nltk_data\tokenizers\punkt\PY3\spanish.pickle')
tokenizer = nltk.data.load('.\Sandra\AppData\Roaming\nltk_data\tokenizers\punkt\PY3\portuguese.pickle')
tokenizer = nltk.data.load('.\Sandra\AppData\Roaming\nltk_data\tokenizers\punkt\PY3\bulgarian.pickle')


#tokenizer type
from nltk.tokenize import word_tokenize

#initialize tokenization
tokenizer = nltk.tokenize.word_tokenize(r'\w+')

#data['content'].head(5)
data['content'].apply(tokenizer.tokenize).head()

#stop words
from nltk.corpus import stopwords
set(stopwords.words('english','spanish','portuguese','turkish', 'bulgarian'))

def remove_stopwords(text):
	words = [w for w in text if w not in stopwords.words('english')]
	return words['content']
	data['content'].apply(remove_stopwords)
