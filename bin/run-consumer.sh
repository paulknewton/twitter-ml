# Read messages from Kafka via Spark and process.
#
# Kafka topic: 'brexit'

# fix problem with multi-threading on Mac
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# load Kafka libs from Maven
SPARK_ARGS="--verbose --conf spark.jars.packages=org.apache.spark:spark-streaming-kafka-0-8-assembly_2.11:2.4.0"

# create a zip archive to distribute python code to clusters
SRC=doc_scanner
SRC_ZIP=$SRC.zip
pushd ..
zip -rFS $SRC_ZIP $SRC -i "*.py"
popd

# virtualenv (with nltk etc)
#VENV_ZIP=zip/virtualenv.zip
#zip -rFS $VENV_ZIP virtualenv

PY_FILES="--py-files /Users/paul/github/spark/$SRC_ZIP"

#PYSPARK_PYTHON=./NLTK/nltk_env/bin/python

PYSPARK_PYTHON=python spark-submit \
$SPARK_ARGS --master local[*] $PY_FILES \
python twitter_ml.kafka.read_tweets_from_kafka.py
