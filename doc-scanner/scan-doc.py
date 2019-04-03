# Analyse a text document and extract key metrics.
#
# Uses basic RDD/list techniques and MapReduce for counting lines/words/unique words.
# Removes common words via the nltk library.
# Plots most frequently used words using pandas matlab plots.
#

import argparse, logging, re

# init logging
logger = logging.getLogger('csvIt')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def todo(logger, s):
    logger.info("*** TODO: " + s)


# connect to a Spark cluster
from pyspark.sql import SparkSession
spark = SparkSession\
        .builder\
        .appName("Doc Scanner")\
        .getOrCreate()

def dump_stats(rdd):
    print("--------------------\nDumping stats\n--------------------")
    #logger.debug('RDD:')
    #logger.debug('\n'.join(rdd.take(10)))
    print('num lines = %d' % rdd.count())

    # drop non-alpha characters
    rdd = rdd.map(lambda x: re.sub("[^a-zA-Z\s]+","", x).lower().strip())
    #logger.debug('RDD WITH ONLY ALPHA CHARS:')
    #logger.debug('\n'.join(rdd.take(10)))

    # split RDD into individual words
    words = rdd.flatMap(lambda x: x.split(" "))
    #logger.debug('RDD INDIV. WORDS:')
    #logger.debug('\n'.join(words.take(10)))
    print("num words = %d" % words.count())

    # drop empty and single letter words
    words = words.filter(lambda x: len(x) > 1)


def drop_stopwords(rdd):
    logger.debug('Dropping stopwords with NLTK')

    # drop stop words (downloads words if needed)
    # If you are behind a proxy, set NLTK_PROXY in your environment
    import nltk, os
    if 'NLTK_PROXY' in os.environ:
        logger.debug('Using proxy %s' % os.environ['NLTK_PROXY'])
        nltk.set_proxy(os.environ['NLTK_PROXY'])
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    words = rdd.filter(lambda x: x not in stopwords)

    # extract pairs (MapReduce)
    word_pairs = words.map(lambda x: (x,1))
    #logger.debug(word_pairs.count())
    #logger.debug(word_pairs.take(10))

    # count by reduction (MapReduce)
    unique_words = word_pairs.reduceByKey(lambda a, b: a + b)
    print('num unique words = %d' % unique_words.count())
    #logger.debug(unique_words.takeOrdered(30, key = lambda x: -x[1]))

    return unique_words


def plot_graph(rdd):
    # plot some graphs (pandas has more control than built-in databricks visualisations)
    words_to_plot = rdd.takeOrdered(80, key = lambda x: -x[1])
    #display(words_to_plot)

    # create a dictionary of data {x,y}
    import pandas as pd
    import matplotlib.pyplot as plt

    logger.debug('Preparing data...')
    words_word = [x[0] for x in words_to_plot]
    words_count = [x[1] for x in words_to_plot]
    words_dict = {"word": words_word, "frequency": words_count}

    # convert to pandas DF and plot
    df_words = pd.DataFrame(words_dict)

    logger.debug('Preparing plot...')
    word_plot = df_words.plot(figsize=(20, 50), x="word", y="frequency", kind="barh", legend=False)
    word_plot.invert_yaxis()
    plt.title("Word Frequency", fontsize=28)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.ylabel("")

    #display(word_plot.figure)
    logger.debug('Saving figure')
    plt.savefig('tmp/word-freq.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Spark program to process text files and analyse contents')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='verbose logging')
    parser.add_argument('file', help='file to process')
    parser.add_argument('-s', dest='drop_stopwords', action='store_true', default=False, help='strip stopwords')
    parser.add_argument('-p', dest='plot', action='store_true', default=False, help='plot figure')
    args = parser.parse_args()
    if args.verbose:
         logger.setLevel(logging.DEBUG)
    logger.debug("args = %s" % args)

    # read the file into an RDD
    logger.info('Processing %s' % args.file)
    rdd = spark.sparkContext.textFile(args.file)
    dump_stats(rdd)

    if args.drop_stopwords:
        rdd = drop_stopwords(rdd)

    if args.plot:
        plot_graph(rdd)
