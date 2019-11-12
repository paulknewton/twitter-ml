"""Module to process tweets from Twitter."""
import json

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from twitter_ml.classify.sentiment import Sentiment

sc = SparkContext(appName="PythonSparkStreamingKafka")
sc.setLogLevel("WARN")

# TODO read from config file
ssc = StreamingContext(sc, 10)
kafkaStream = KafkaUtils.createStream(
    ssc, "kafka:2181", "sentiment-analyser", {"brexit": 1}
)

classifier = Sentiment("voting.yaml")


def process_tweet(tweet: str):
    """
    Read a tweet and classify it.

    :param tweet: the twitter tweet as a JSON structure
    """
    tweet_text = json.loads(tweet)["text"]
    # print(tweet_text)

    category = classifier.classify_sentiment(tweet_text)

    print("----------------\n%s\n>>>>>> %s" % (tweet_text, category))


kafkaStream.foreachRDD(lambda rdd: rdd.foreach(lambda row: process_tweet(row[1])))

# start the Spark streaming
ssc.start()
ssc.awaitTermination()
