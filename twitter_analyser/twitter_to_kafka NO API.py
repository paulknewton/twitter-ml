# Simple Twitter producer for Kafka
#
# Listens for tweets containing keywords (see 'track=...') and publishes to a Kafka topic ('send_messages("xxx",...)
#

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import SimpleProducer, KafkaClient
import argparse

# Twitter keys. Get these from the Twitter Dev portal.
access_token = "ACCESS_TOKEN"
access_token_secret = "ACCESS_TOKEN_SECRET"
consumer_key =  "CONSUMER_API_KEY"
consumer_secret =  "CONSUMER_SECRET_KEY"

publish_topic = "brexit"  # topic used to publish

class StdOutListener(StreamListener):
    def on_data(self, data):
        producer.send_messages(publish_topic, data.encode('utf-8'))
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == "__main__":
    # read command-line args
    parser = argparse.ArgumentParser(description='Consumer to read tweets from Kafka and classify sentiment.')
    parser.add_argument('--twitter', nargs=1, dest='twitter', default="brexit",
                        help='topic to extract from Twitter')
    args = parser.parse_args()

    # kafka connection
    kafka = KafkaClient("kafka:9092")
    producer = SimpleProducer(kafka)

    # twitter connection
    twitter_topic = args.twitter[0]
    print("Reading tweets for topic '%s'" % twitter_topic)
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=[twitter_topic])
