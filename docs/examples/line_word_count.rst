Find lines with the most words
==============================

This is an example based on the Getting Started examples
(https://spark.apache.org/docs/latest/quick-start.html) from the Apache
site. The sample finds the line(s) in a text file with the most number
of words.

The example does not use a MapReduce model, but shows the basic
DataFrame API which is the main data abstraction in Spark.

Start by reading a file into a DataFrame (via the SparkContext). Return
a line count and show the first row.

::

   from pyspark.sql import SparkSession

   # create a SparkContext. If running in DataBricks, this is already done for you
   spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

   filename = "FileStore/tables/README.md"
   filename = "README.md"

   textFile = spark.read.text(filename)
   textFile.count() 
   textFile.first()

Now split each line into an array of words using the split function.
This array of words can be counted to show the number of words on each
line.

::

   from pyspark.sql.functions import *

   # creates a column of arrays (array is spark.sql function split using regexp)
   words = split(textFile.value, "\s+").name("words")

   # create a DF containing a single column of word counts (size is built-in to count the num elements in each Column item)
   wcDf = textFile.select(size(words).name("numWords"), words)
   #wcDf.show()

   # find max value in the Column and convert to a DF
   wcDf.agg(max(col("numWords"))).show()

This outputs:

::

   +-------------+
   |max(numWords)|
   +-------------+
   |           94|
   +-------------+

We can use this to extract the actual rows that match this word count.
This uses the filter method to find matching records.

.. code:: python

   # show rows with maxWords (this requires 2 passes over the DF - not efficient). Note filter syntax uses column notation.
   maxValue = wcDf.agg(max(col("numWords"))).collect()[0][0]
   wcDf.filter(wcDf.numWords == maxValue).show()

This outputs:

::

   +--------+--------------------+
   |numWords|               words|
   +--------+--------------------+
   |      94|[Spark, and, the,...|
   +--------+--------------------+
