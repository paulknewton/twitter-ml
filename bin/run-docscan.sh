# Run the doc-scanner job
#

cd ~/tools/spark
bin/spark-submit ~/github/twitterML/docscan.py --stopwords --plot -v "$@" /Users/paul/github/twitterML/sample-text.txt
