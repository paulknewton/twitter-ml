# Getting Started
* The Apache site is a good place to start (https://spark.apache.org/), especially the [Getting Started](https://spark.apache.org/docs/latest/quick-start.html) section. Explains the concepts, how to install, and lots of examples.
* Databricks publish a shortprimer for developers (https://pages.databricks.com/7-steps-for-a-developer-to-learn-apache-spark.html). Nothing too detailed, but gives a rough idea of the programming concepts and API

# Examples
* Apache have some examples to get you started (https://spark.apache.org/examples.html). e.g. the classic 'word count using MapReduce' or 'calculate Pi by choosing random points'. These are good introductions to using the Spark API (and functional programming in general).
* The Spark code also comes with many examples (this is the Python link to the Git repo: https://github.com/apache/spark/tree/master/examples/src/main/python)
* These are my annotated Python examples: either taken from other sources or invented by me (https://github.com/paulknewton/spark/blob/master/examples-python.md). They include explanations of how the code is working, which data types are used etc. I noted these down when I was working through the examples - they helped me, so I hope they help you too.

# Hosted Spark
You can run Spark in many different modes - either locally as a standlone installation, or - more usefully - as as cluster to perform parallel computation. All the instructions are on the Apache site.

But why not use a cloud provider to host Spark for you? Lets you focus on the programming instead of administering a Spark installation.

* Databricks (https://databricks.com/)
    * Launched by the original authors of Apache Spark.
    * Offers a free Community edition with limited storage (https://community.cloud.databricks.com)
    * Supports Jupiter Notebooks for interactive coding and execution on clusters.

# Research
These are some useful sections on the more theoretical ideas that you find in Spark.
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
