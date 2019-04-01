
# Annotated examples in Python
These are based on the Getting Started (https://spark.apache.org/docs/latest/quick-start.html) examples from the  Apache site.

## Find lines with the most words

Start by reading a file into a DataFrame (via the SparkContext)

```
from pyspark.sql import SparkSession

# create a SparkContext. If running in DataBricks, this is already done for you
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

filename = "FileStore/tables/README.md"
filename = "README.md"

textFile = spark.read.text(filename)
textFile.count() 
textFile.first()
```
Now split the file into rows.
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

+-------------+
|max(numWords)|
+-------------+
|           94|
+-------------+



We can print the calculated maxValue.
We can also use this to extract the actual rows that match this word count.
This uses the filter method to find matching records.


```python
# show rows with maxWords (this requires 2 passes over the DF - not efficient). Note filter syntax uses column notation.
maxValue = wcDf.agg(max(col("numWords"))).collect()[0][0]
wcDf.filter(wcDf.numWords == maxValue).show()
```

+--------+--------------------+
|numWords|               words|
+--------+--------------------+
|      94|[Spark, and, the,...|
+--------+--------------------+


