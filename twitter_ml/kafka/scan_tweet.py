import json

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

from twitter_ml.classify.build_classifiers import classify_sentiment

sc = SparkContext(appName="PythonSparkStreamingKafka")
sc.setLogLevel("WARN")

ssc = StreamingContext(sc, 10)
kafkaStream = KafkaUtils.createStream(ssc, 'kafka:2181', 'sentiment-analyser', {'brexit': 1})


def processTweet(tweet):
    tweet_text = json.loads(tweet)["text"]
    # print(tweet_text)

    (sentiment, confidence) = classify_sentiment(tweet_text, False)

    print("----------------\n%s\n>>>>>> %s (%f)" % (tweet_text, sentiment, confidence))


kafkaStream.foreachRDD(lambda rdd: rdd.foreach(lambda row: processTweet(row[1])))

# start the Spark streaming
ssc.start()
ssc.awaitTermination()
