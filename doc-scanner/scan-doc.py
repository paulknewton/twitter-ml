# Analyse a text document and extract key metrics.
#
# Uses basic RDD/list techniques and MapReduce for counting lines/words/unique words.
# Removes common words via the nltk library.
# Plots most frequently used words using pandas matlab plots.
#

# connect to a Spark cluster
from pyspark.sql import SparkSession
spark = SparkSession\
        .builder\
        .appName("Doc Scanner")\
        .getOrCreate()

# read the file into an RDD
rdd = spark.sparkContext.textFile("tmp/brexit.txt")
#print('\n'.join(rdd.take(10)))
print("#lines = %d" % rdd.count())

# drop non-alpha characters
import re
rdd = rdd.map(lambda x: re.sub("[^a-zA-Z\s]+","", x).lower().strip())
#print('\n'.join(rdd.take(10)))

# split RDD into individual words
words = rdd.flatMap(lambda x: x.split(" "))
#print('\n'.join(words.take(10)))
print("#words = %d" % words.count())

# drop empty and single letter words
words = words.filter(lambda x: len(x) > 1)

# drop stop words (downloads words if needed)
# If you are behind a proxy, set NLTK_PROXY in your environment
import nltk, os
if 'NLTK_PROXY' in os.environ:
    print('Using proxy %s' % os.environ['NLTK_PROXY'])
    nltk.set_proxy(os.environ['NLTK_PROXY'])
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
words = words.filter(lambda x: x not in stopwords)

# extract pairs (MapReduce)
word_pairs = words.map(lambda x: (x,1))
#print(word_pairs.count())
#print(*word_pairs.take(10), sep = "\n")

# count by reduction (MapReduce)
unique_words = word_pairs.reduceByKey(lambda a, b: a + b)
print('#unique words = %d' % unique_words.count())
#print(*unique_words.takeOrdered(30, key = lambda x: -x[1]), sep = "\n")

# plot some graphs (pandas has more control than built-in databricks visualisations)
words_to_plot = unique_words.takeOrdered(80, key = lambda x: -x[1])
#display(words_to_plot)

# create a dictionary of data {x,y}
import pandas as pd
import matplotlib.pyplot as plt
words_word = [x[0] for x in words_to_plot]
words_count = [x[1] for x in words_to_plot]
words_dict = {"word": words_word, "frequency": words_count}

# convert to pandas DF and plot
df_words = pd.DataFrame(words_dict)
word_plot = df_words.plot(figsize=(20, 50), x="word", y="frequency", kind="barh", legend=False)
word_plot.invert_yaxis()
plt.title("Word Frequency", fontsize=28)
plt.xticks(size=18)
plt.yticks(size=18)
plt.ylabel("")

#display(word_plot.figure)
plt.savefig('tmp/word-freq.png')
