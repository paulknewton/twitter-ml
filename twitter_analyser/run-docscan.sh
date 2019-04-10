# Run the doc-scanner job
#

cd ~/tools/spark
bin/spark-submit ~/github/spark/doc-scanner/docscan.py --stopwords --plot -v "$@" /Users/paul/github/spark/doc-scanner/sample-text.txt
