=====
Usage
=====

Document Scanner
----------------

Start the analysis job (SPARK_ROOT is the folder where you installed Spark; path-to-this-git-repo is the place you cloned this repository):

.. code-block::

    cd $SPARK_ROOT
    bin/spark-submit path-to-this-git-repo/doc-scanner/scan-doc.py some-file-to-analyse


The program supports a number of command line arguments:

.. code-block::

    usage: scan-doc.py [-h] [-v] [-s] [-p] file

    Spark program to process text files and analyse contents

    positional arguments:
      file        file to process

    optional arguments:
      -h, --help  show this help message and exit
      -v          verbose logging
      -s          strip stopwords
      -p          plot figure

Twitter-Kafka Publisher
-----------------------
The twitter client needs API keys to read from Twitter. Sign-up on the `Twitter <https://www.twitter.com>`_ developer platform to get your own keys. Insert your API keys into the code.

* Start by running Zookeeper:

.. code-block::

    bin/zookeeper-server-start.sh config/zookeeper.properties

* Start the Kafka server:

.. code-block::

    bin/kafka-server-start.sh config/server.properties

* Create a Kafka topic (we only need to do this once):

.. code-block::

    bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic brexit
    bin/kafka-topics.sh --list --bootstrap-server localhost:9092

* Start the console listener (this is just to check Kafka is receiving tweets):

.. code-block::

    bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic brexit --from-beginning

* Start the Twitter producer:

.. code-block::

    python twitter-to-kafka.py

This will read tweets from Twitter and pump them into Kafka. It will also print the tweets to the console.

The Twitter Analyser
--------------------
I had to define a variable to enable multi-threaded applications on a Mac (apparently due to `security changes <https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr>`_:

.. code-block::

    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

* Start the analysis job (SPARK_ROOT is the folder where you installed Spark; path-to-this-git-repo is the place you cloned this repository):

.. code-block::

    cd $SPARK_ROOT
    bin/spark-submit path-to-this-git-repo/twitter-stream-analyser/read-tweets-kafka.py

This will launch the Spark platform in standalone mode and submit the python job.
This job reads tweets from Kafka.

Running from PyCharm
--------------------
`This blog <https://www.pavanpkulkarni.com/blog/12-pyspark-in-pycharm/>`_ has some useful information on running Spark jobs from PyCharm.

In summary:

* Edit your ``.profile`` (or ``.bash_profile``, or whatever) to add the ``SPARK_HOME`` and ``PYTHONPATH`` settings)
* Add the Hadoop python libraries to the PyCharm project interpreter settings
* Edit ``$SPARK_HOME/conf/spark-default.conf`` to include the line:

.. code-block::

    spark.jars.packages org.apache.spark:spark-streaming-kafka-0-8-assembly_2.11:2.4.0

Note: the actual version settings depend on the version of Spark (2.4.0), the version of Scala (2.11) and Kafka.
If you try running your Spark program, it will print an error message that tells you which version to add.
This will be used to download the relevent JARs from Maven the first time you run the code.
