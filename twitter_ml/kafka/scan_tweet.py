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

classifier = Sentiment()


def process_tweet(tweet):
    tweet_text = json.loads(tweet)["text"]
    # print(tweet_text)

    sentiment, confidence = classifier.classify_sentiment(tweet_text)

    print("----------------\n%s\n>>>>>> %s (%f)" % (tweet_text, sentiment, confidence))


kafkaStream.foreachRDD(lambda rdd: rdd.foreach(lambda row: process_tweet(row[1])))

# start the Spark streaming
ssc.start()
ssc.awaitTermination()
