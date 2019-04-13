"""
Analyse a text document and extract key metrics.

Uses basic RDD/list techniques and MapReduce for counting lines/words/unique words.
Removes common words via the nltk library.
Plots most frequently used words using pandas matlab plots.
"""

import argparse, logging, re, sys

# init logging
logger = logging.getLogger('doc_scanner')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def todo(logger, s):
    logger.info("*** TODO: " + s)


# connect to a Spark cluster
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Doc Scanner") \
    .getOrCreate()


def flatten_text(rdd):
    """Split the RDD into individual words (1 per row) and return the transformed RDD."""

    # logger.debug('RDD:')
    # logger.debug('\n'.join(rdd.take(10)))
    print('num lines = %d' % rdd.count())

    # drop non-alpha characters
    rdd = rdd.map(lambda x: re.sub("[^a-zA-Z\s]+", "", x).lower().strip())
    # logger.debug('RDD WITH ONLY ALPHA CHARS:')
    # logger.debug('\n'.join(rdd.take(10)))

    # split RDD into individual words
    words = rdd.flatMap(lambda x: x.split(" "))
    # logger.debug('RDD INDIV. WORDS:')
    # logger.debug('\n'.join(words.take(10)))
    print('num words = %d' % words.count())

    return words


def drop_stopwords(rdd):
    """Remove commonly occurring 'stop' words from the flattenned RDD and return the new RDD."""

    print('num words with stop words = %d' % rdd.count())

    # drop empty and single letter words
    words = rdd.filter(lambda x: len(x) > 1)
    print('num words with short words removed = %d' % words.count())

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
    words = words.filter(lambda x: x not in stopwords)
    print('num words without stop words = %d' % words.count())

    return words


def filter_unique(rdd):
    """Count occurrences of each word and return an RDD of (word,count) pairs."""

    print('num words non-unique = %d' % rdd.count())

    # extract pairs (MapReduce)
    word_pairs = rdd.map(lambda x: (x, 1))
    # logger.debug(word_pairs.count())
    # logger.debug(word_pairs.take(10))

    # count by reduction (MapReduce)
    unique_words = word_pairs.reduceByKey(lambda a, b: a + b)
    print('num unique words = %d' % unique_words.count())
    # logger.debug(unique_words.takeOrdered(30, key = lambda x: -x[1]))

    return unique_words


def plot_graph(rdd):
    """Save a MATLAB-style figure of an RDD with the form (word, freq)."""

    # plot some graphs (pandas has more control than built-in databricks visualisations)
    words_to_plot = rdd.takeOrdered(80, key=lambda x: -x[1])
    # display(words_to_plot)

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

    # display(word_plot.figure)
    logger.debug('Saving figure')
    plt.savefig('word-freq.png')


def calc_sentiment(rdd):
    """Calculate the sentiment of the text."""

    raise NotImplementedError("To be implemented")
    import stanfordnlp
    stanfordnlp.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanfordnlp.Pipeline()  # This sets up a default neural pipeline in English
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc.sentences[0].print_dependencies()


if __name__ == "__main__":

    # read command-line args
    parser = argparse.ArgumentParser(description='Spark program to process text files and analyse contents')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='verbose logging')
    parser.add_argument('file', help='file to process')
    parser.add_argument('--stopwords', dest='drop_stopwords', action='store_true', default=False, help='strip stopwords')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='plot figure')
    parser.add_argument('--sentiment', dest='sentiment', action='store_true', default=False, help='activate sentiment anlaysis')
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("args = %s" % args)

    if args.sentiment:
        calc_sentiment(None)
        sys.exit()

    # read the file into an RDD
    logger.info('Processing %s' % args.file)
    rdd = spark.sparkContext.textFile(args.file)

    rdd = flatten_text(rdd)

    if args.drop_stopwords:
        rdd = drop_stopwords(rdd)

    rdd = filter_unique(rdd)

    if args.plot:
        plot_graph(rdd)
