# Getting Started
* The Apache site is a good place to start (https://spark.apache.org/), especially the [Getting Started](https://spark.apache.org/docs/latest/quick-start.html) section. Explains the concepts, how to install, and lots of examples.
* Databricks publish a shortprimer for developers (https://pages.databricks.com/7-steps-for-a-developer-to-learn-apache-spark.html). Nothing too detailed, but gives a rough idea of the programming concepts and API

# Running Spark
* The Apache site has [instructions](https://spark.apache.org/downloads.html) on downloading and installing Spark
* If your focus is on learning to program with Spark, I would strongly recommend using one of the cloud providers that offer a hosted installation of Spark with Jupyter notebooks. This means you don't install anything locally. Just sign-up on-line and type your programs into the web-based notebook. This is so easy.
    * Databricks (https://databricks.com/)
        * Launched by the original authors of Apache Spark.
        * Offers a free Community edition with limited storage (https://community.cloud.databricks.com)
        * Supports Jupiter Notebooks for interactive coding and execution on clusters.
        * This is the one I have been using. I don't if it is better than the others, my experiences have been very positive so far - simple, reliable and does what I need.
* If you really want to run a Spark installation locally, I would suggest going the [docker](https://spark.apache.org/downloads.html) route. Just run a single docker command line to pull a pre-prepared image with all dependencies.
    * [This](https://github.com/CoorpAcademy/docker-pyspark) is a vanilla Spark image ready for use (Spark 2.0)
    * [This](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html) is an image containing Spark and Jupyter notebooks. These notebooks are great! Let's you run your programs interactively by typing them into a notebook. Gives you roughly the same experience as the Databricks example above. Here is a [tutorial](https://levelup.gitconnected.com/using-docker-and-pyspark-134cd4cab867) on installing the image if you prefer.
* If you want to build a real-world hardware cluster, you can run Spark on a cluster of Raspberry Pi computers. Do not expect lightning performance, but this is a one way of actually building a cluster of machines to deploy a Spark network. Very interesting. You can find quite a few links. Try [here](http://fisheyefocus.com/fisheyeview/?p=548) or [here](http://bailiwick.io/2015/07/07/create-your-own-apache-spark-cluster-using-raspberry-pi-2/)

# Examples
* Apache have some examples to get you started (https://spark.apache.org/examples.html). e.g. the classic 'word count using MapReduce' or 'calculate Pi by choosing random points'. These are good introductions to using the Spark API (and functional programming in general).
* The Spark code also comes with many examples (this is the Python link to the Git repo: https://github.com/apache/spark/tree/master/examples/src/main/python)
* These are my annotated Python examples: either taken from other sources or invented by me (https://github.com/paulknewton/spark/blob/master/examples-python.md). They include explanations of how the code is working, which data types are used etc. I noted these down when I was working through the examples - they helped me, so I hope they help you too.

# Streaming
Spark and the other platforms like Hadoop have their origins in batch processing - crunching large data sets over hours and hours, and a magic result pops out the other end. But this doesn't fit all of our use cases - data is often continuous and needs to be processed in real-time. Spark provides a complete streaming package to support this. And most importantly, it uses (almost) the same programming model. This makes life much easier for developers. The more I read about streaming, the more I think this is where Spark really shines.

* Start with the excellent [programming guide](https://spark.apache.org/docs/latest/streaming-programming-guide.html) on the Apache site
* There are a number of tutorials and examples on-line. I like [this one](https://prateekvjoshi.com/2015/12/22/analyzing-real-time-data-with-spark-streaming-in-python/) which reads a data stream from a socket and classifies it in real-time. Simple. Clear. And shows how to use lamba functions.
* Here is another one from [Databricks](https://databricks.com/spark/getting-started-with-apache-spark/streaming) which is the hosting platform I have been playing with. A more complex example that reads JSON event data from files.

# Research & Random Ideas
These are some useful sections on the more theoretical ideas that you find in Spark.

## Functional Programming
Some random bits and pieces on functional programming and related stuff. Not necessarily specific to Spark or Big Data, but useful.
* [Monkey patching](https://www.geeksforgeeks.org/monkey-patching-in-python-dynamic-behavior/) - allows you to dynamically change runtime behaviour in Python
* [Currying](https://en.wikipedia.org/wiki/Currying) - transforming multi-argument functions to a sequence of single-argument functions. Some [Python-specific](https://www.python-course.eu/currying_in_python.php) examples.
* [Chaining DataFrame transformations](https://medium.com/@mrpowers/chaining-custom-pyspark-transformations-4f38a8c7ae55) - how to define transformation functions to they can be chained as a sequence. Avoids lots of redundant DataFrame variables.
* [Reducing DataFrames](https://medium.com/@mrpowers/performing-operations-on-multiple-columns-in-a-pyspark-dataframe-36e97896c378) - different solutions for apply reduce functions to DataFrames

## Machine Learning
* Spark enables computation and analysis of large data sets, so is well-suited to the area of machine learning. Spark includes a machine learning framework called [Mlib](https://spark.apache.org/mllib/).
* A full university course online! CS229: Machine Learning (http://cs229.stanford.edu/ and https://see.stanford.edu/Course/CS229). Full download of course material and on-line videos. This is a full course from Stanford University, so be prepared to invest a significant amount of time. And yes, you will need a reasonable level of mathematics.
* Introduction to Machine Learning from Google (https://developers.google.com/machine-learning/crash-course/ml-intro)

# Books
There are more and more books on the subject of Spark. This is a good sign - if publishers are willing to invest in these titles, it suggests strong reader-demand. I can only include comments on books that I have actually read (or at least looked at), but here are my thoughts:
* [Learning Spark: Lightning-Fast Big Data Analysis By Matei Zaharia, Holden Karau, Andy Konwinski, Patrick Wendell](http://shop.oreilly.com/product/0636920028512.do). I am reading this now. Looks like a good introduction to the platform, straight from some of the original development team.

# Certification
Don't think about this when you are getting started. Certifications can be helpful to give you a target and provide some structure for learning. Yes, they can show others (including employers) that you have a certain skillset. But be careful of using these as an end in themselves.

Many providers offer some certification programme. e.g.
* Databricks Certified Apache Spark Developer (https://databricks.com/training/certified-spark-developer). Cost: $300
