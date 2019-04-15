"""
Analyse a text document and extract key metrics.

Uses basic RDD/list techniques and MapReduce for counting lines/words/unique words.
Removes common words via the nltk library.
Plots most frequently used words using pandas matlab plots.
"""

import argparse, logging, re, sys
import pandas as pd
import matplotlib.pyplot as plt

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
    .appName("Doc Scanner").master("local").config("spark.driver.host", "localhost") \
    .getOrCreate()


def flatten_text(rdd, stats):
    """Split the RDD into individual words (1 per row) and return the transformed RDD."""

    # logger.debug('RDD:')
    # logger.debug('\n'.join(rdd.take(10)))
    stats.append(("num lines", rdd.count()))

    # drop non-alpha characters
    rdd = rdd.map(lambda x: re.sub("[^a-zA-Z\s]+", "", x).lower().strip())
    # logger.debug('RDD WITH ONLY ALPHA CHARS:')
    # logger.debug('\n'.join(rdd.take(10)))

    # split RDD into individual words
    words = rdd.flatMap(lambda x: x.split(" "))
    # logger.debug('RDD INDIV. WORDS:')
    # logger.debug('\n'.join(words.take(10)))
    stats.append(("num words", words.count()))

    return words


def drop_stopwords(rdd, stats):
    """Remove commonly occurring 'stop' words from the flattenned RDD and return the new RDD."""

    stats.append(("num words with stop words", rdd.count()))

    # drop empty and single letter words
    words = rdd.filter(lambda x: len(x) > 1)
    stats.append(("num words with short words removed", words.count()))

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
    stats.append(("num words without stop words", words.count()))

    return words


def filter_unique(rdd, stats):
    """Count occurrences of each word and return an RDD of (word,count) pairs."""

    stats.append(("num words non-unique", rdd.count()))

    # extract pairs (MapReduce)
    word_pairs = rdd.map(lambda x: (x, 1))
    # logger.debug(word_pairs.count())
    # logger.debug(word_pairs.take(10))

    # count by reduction (MapReduce)
    unique_words = word_pairs.reduceByKey(lambda a, b: a + b)
    stats.append(("num unique words", unique_words.count()))
    # logger.debug(unique_words.takeOrdered(30, key = lambda x: -x[1]))

    return unique_words


def print_stats(stats):
    for s in stats:
        print(s)


def plot_stats(stats):
    df = pd.DataFrame.from_dict({"statistic": [stat[0] for stat in stats], "summary": [stat[1] for stat in stats]})
    df.set_index("statistic", inplace=True)
    df.plot(kind="barh", grid=True)
    plt.tight_layout()

    plt.savefig("fig_summary.png")
    plt.show()


def plot_word_freq(rdd):
    # plot some graphs
    words_to_plot = rdd.takeOrdered(80, key=lambda x: -x[1])

    logger.debug('Preparing data...')
    words_word = [x[0] for x in words_to_plot]
    words_count = [x[1] for x in words_to_plot]
    words_dict = {"word": words_word, "frequency": words_count}

    # convert to pandas DF and plot
    df_words = pd.DataFrame(words_dict)

    logger.debug('Preparing plot...')
    word_plot = df_words.plot(figsize=(20, 50), x="word", y="frequency", kind="barh", legend=False, grid=True)
    word_plot.invert_yaxis()
    plt.title("Word Frequency", fontsize=20)
    plt.xticks(size=8)
    plt.yticks(size=8)
    plt.ylabel("")

    plt.savefig('fig_word_freq.png')
    plt.show()


if __name__ == "__main__":

    # read command-line args
    parser = argparse.ArgumentParser(description='Spark program to process text files and analyse contents')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='verbose logging')
    parser.add_argument('file', help='file to process')
    parser.add_argument('--stopwords', dest='drop_stopwords', action='store_true', default=False,
                        help='strip stopwords')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='plot figure')
    parser.add_argument('--sentiment', dest='sentiment', action='store_true', default=False,
                        help='activate sentiment anlaysis')
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

    stats = []  # list of tuples (description, value)
    rdd = flatten_text(rdd, stats)

    if args.drop_stopwords:
        rdd = drop_stopwords(rdd, stats)

    rdd = filter_unique(rdd, stats)

    # print_stats(stats)

    if args.plot:
        # import matplotlib
        # font = {'size': 6}
        # matplotlib.rc('font', **font)

        plot_stats(stats)
        plot_word_freq(rdd)
