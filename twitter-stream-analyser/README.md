# Twitter Stream Analyser

## Running the Kafka server

Our Twitter feed uses Apache Kafka to publish the stream of tweets.
Download it from [here](https://kafka.apache.org/). Unpack the tarball.

Start by running Zookeeper:
```
bin/zookeeper-server-start.sh config/zookeeper.properties
```

Start the Kafka server:
```
bin/kafka-server-start.sh config/server.properties

```

Create a Kafka topic (we only need to do this once):
```
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic brexit
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

Start the console listener:
``` 
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic brexit --from-beginning
```

## Running the Twitter client
I suggest creating a python virtualenv.
Install the python libraries:

```
pip install kafka-python
pip install python-twitter
pip install tweepy
```

Start the Twitter feed:
```
python twitter-to-kafka.py
```
