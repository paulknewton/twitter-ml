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
This code reads tweets from Twitter and pushes them into Kafa on a specific topic. This is the data producer part...

You will need to install some python dependencies. I suggest creating a python virtualenv first, then install the python libraries:

```
pip install kafka-python
pip install python-twitter
pip install tweepy
```

The twitter client needs API keys to read from Twitter. Sign-up on the Twitter developer platform to get your own keys.
Rename twitter-to-kafka NO API.py to twitter-to-kafka.py and insert your API keys.

Start the Twitter feed:
```
python twitter-to-kafka.py
```
This will read tweets from Twitter and pump them into Kafka. It will also print the tweets to the console.

## Running the Twitter analyser
This code reads tweets from Kafka and performs analyse of the text using the Spark platform.

You will need spark installed/downloaded on your machine to run this.
Start the analysis job (SPARK_ROOT is the folder where you installed Spark; path-to-this-git-repo is the place you cloned this repository):
```
cd $SPARK_ROOT
bin/spark-submit path-to-this-git-repo/twitter-stream-analyser/read-tweets-kafka.py
```

This will launch the Spark platform in standalone mode and submit the python job.
This job reads tweets from Kafka.
