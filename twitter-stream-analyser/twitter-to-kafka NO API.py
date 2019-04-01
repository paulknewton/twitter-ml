# Simple Twitter producer for Kafka
#
# Listens for tweets containing keywords (see 'track=...') and publishes to a Kafka topic ('send_messages("xxx",...)
#

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import SimpleProducer, KafkaClient

# Twitter keys. Get these from the Twitter Dev portal.
access_token = "ACCESS_TOKEN"
access_token_secret = "ACCESS_TOKEN_SECRET"
consumer_key =  "CONSUMER_API_KEY"
consumer_secret =  "CONSUMER_SECRET_KEY"

class StdOutListener(StreamListener):
    def on_data(self, data):
        producer.send_messages("brexit", data.encode('utf-8'))
        print (data)
        return True
    def on_error(self, status):
        print (status)

kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka)
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)
stream.filter(track=["brexit"])
