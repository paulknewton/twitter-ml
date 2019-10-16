import argparse
import logging.config

import yaml
from kafka import SimpleProducer, KafkaClient
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

from twitter_ml.utils.config import Config

publish_topic = "brexit"  # topic used to publish

with open("logging.yml", 'rt') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))

logger = logging.getLogger(__name__)


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

    config = Config("twitter.yml")

    # read twitter setup
    twitter_config = config.get_config_value("twitter", config.root)
    if not twitter_config:
        raise KeyError("Could not find section 'frame' in config file. Exiting")

    twitter_token = config.get_config_value("twitter_token", twitter_config)
    twitter_secret = config.get_config_value("twitter_secret", twitter_config)
    consumer_key = config.get_config_value("consumer_key", twitter_config)
    consumer_secret = config.get_config_value("consumer_secret", twitter_config)

    twitter_topic = args.twitter
    print("Reading tweets for topic '%s'" % twitter_topic)

    # kafka connection
    kafka = KafkaClient("kafka:9092")
    producer = SimpleProducer(kafka)

    output = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, output)
    stream.filter(track=[twitter_topic])
