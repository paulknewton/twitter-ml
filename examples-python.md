# Annotated examples in Python
These are based on the Getting Started (https://spark.apache.org/docs/latest/quick-start.html) examples from the  Apache site.

## Find lines with the most words

Read a file into a DataFrame (via the SparkContext)
```
textFile = spark.read.text("FileStore/tables/README.md")
textFile.count() 
textFile.first()
```

Count words in a file.
This uses built-in functions from the spark.sql module. Each function returns a Column. Note how the select function is used to transform this to a new DataFrame.
```
from pyspark.sql.functions import *
#textFile.select(size(split(textFile.value, "\s+")).name("numWords")).agg(max(col("numWords"))).collect()

# creates a column of arrays (array is spark.sql function split using regexp)
words = split(textFile.value, "\s+").name("words")

# create a DF containing a single column of word counts (size is built-in to count the num elements in each Column item)
wcDf = textFile.select(size(words).name("numWords"), words)
#wcDf.show()

# find max value in the Column and convert to a DF
wcDf.agg(max(col("numWords"))).show()
```

We can print the calculated maxValue.
We can also use this to extract the actual rows that match this word count.
This uses the filter method to find matching records.
```
# show rows with maxWords (this requires 2 passes over the DF - not efficient). Note filter syntax uses column notation.
maxValue = wcDf.agg(max(col("numWords"))).collect()[0][0]
wcDf.filter(wcDf.numWords == maxValue).show()
```
