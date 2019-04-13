# Twitter Stream Analyser
This is a project to analyse text streams - either standlone documennts or Twitter feeds - and performs various forms of textual analysis. Uses Apache Spark to built textual metrics, then processes the text via various classification models analysis to evalute the sentiment (models via SciKit-Learn).

## The Document Scanner
This is a standlone program that reads a text document and analyses it using NLTK and Spark.

### Dependencies
*Apache Spark*

You will need spark installed/downloaded on your machine to run this.
Note: when running on a Mac, I had to (downgrade) to Java v8 to get Spark to run (otherwise even the Spark examples fail).

*Python*

A few non-standard python libraries may need to be installed:
```
pip install nltk
pip install pandas
pip install matplotlib
pip install corenlp-python
```

### Starting the program
Start the analysis job (SPARK_ROOT is the folder where you installed Spark; path-to-this-git-repo is the place you cloned this repository):
```
cd $SPARK_ROOT
bin/spark-submit path-to-this-git-repo/doc-scanner/scan-doc.py some-file-to-analyse
```

The program supports a number of command line arguments:
```
usage: scan-doc.py [-h] [-v] [-s] [-p] file

Spark program to process text files and analyse contents

positional arguments:
  file        file to process

optional arguments:
  -h, --help  show this help message and exit
  -v          verbose logging
  -s          strip stopwords
  -p          plot figure
```


## Twitter-Kafka Producer
This reads tweets from Twitter and pumps them into a Kafka server (where they can be consumed by out Twitter Consumer programs).

### Dependencies
*Apache Kafka*

Our Twitter feed uses Apache Kafka to publish the stream of tweets.
Download it from [here](https://kafka.apache.org/). Unpack the tarball.

*Python*

You will need to install some more python dependencies. I suggest creating a python virtualenv first, then install the python libraries:

```
pip install kafka-python
pip install python-twitter
pip install tweepy
```

*Twitter API*

he twitter client needs API keys to read from Twitter. Sign-up on the Twitter developer platform to get your own keys.
Rename twitter-to-kafka NO API.py to twitter-to-kafka.py and insert your API keys.

### Starting the program - the Twitter Producer
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

Start the console listener (this is just to check Kafka is receiving tweets):
``` 
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic brexit --from-beginning
```

Start the Twitter producer:
```
python twitter-to-kafka.py
```
This will read tweets from Twitter and pump them into Kafka. It will also print the tweets to the console.

## Starting the program - the Twitter Analyser
This code reads tweets from Kafka and performs analyse of the text using the Spark platform.

You will need spark installed/downloaded on your machine to run this.
Note: when running on a Mac, I had to (downgrade) to Java v8 to get Spark to run (otherwise even the Spark examples fail).
I also had to define a variable to enable multi-threaded applications on a Mac (apparently due to [security changes](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr):
```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

Start the analysis job (SPARK_ROOT is the folder where you installed Spark; path-to-this-git-repo is the place you cloned this repository):

```
cd $SPARK_ROOT
bin/spark-submit path-to-this-git-repo/twitter-stream-analyser/read-tweets-kafka.py
```

This will launch the Spark platform in standalone mode and submit the python job.
This job reads tweets from Kafka.

## Running from PyCharm
[This blog](https://www.pavanpkulkarni.com/blog/12-pyspark-in-pycharm/) has some useful information on running Spark jobs from PyCharm.
In summary:
* Edit your .profile (or .bash_profile, or whatever) to add the SPARK_HOME and PYTHONPATH settings)
* Add the Hadoop python libraries to the PyCharm project interpreter settings
* Install the Twitter python libraries above (tweepy, kafka-python...)
* Edit $SPARK_HOME/conf/spark-default.conf to include the line:
```
spark.jars.packages org.apache.spark:spark-streaming-kafka-0-8-assembly_2.11:2.4.0
```
Note: the actual version settings depend on the version of Spark (2.4.0), the version of Scala (2.11) and Kafka.
If you try running your Spark program, it will print an error message that tells you which version to add.
This will be used to download the relevent JARs from Maven the first time you run the code.

