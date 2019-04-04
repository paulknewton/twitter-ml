# Scanning documents using Spark

## Dependencies

Project to read text documents and perform textual analysis using Spark.

Start by installing the python dependencies:
```
pip install nltk
pip install pandas
pip install matplotlib
pip install corenlp-python
```

## Running the program
You will need spark installed/downloaded on your machine to run this.
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