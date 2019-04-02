import sys, requests, requests_oauthlib, json

# Twitter app config
ACCESS_TOKEN = 'EDIT THIS'
ACCESS_SECRET = 'EDIT THIS'
CONSUMER_KEY = 'EDIT THIS'
CONSUMER_SECRET = 'EDIT THIS'
my_auth = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)

def get_tweets():
  url = 'https://stream.twitter.com/1.1/statuses/filter.json'
  query_data = [('language', 'en'), ('locations', '-130,-20,100,50'),('track','#')]
  query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
  response = requests.get(query_url, auth=my_auth, stream=True)
  print(query_url, response)
  return response

def parse_tweets(http_resp):
  iter = http_resp.iter_lines()
  rdd = spark.sparkContext.parallelize([])
  
  for i in range(0,50):
    line = next(iter).decode('UTF-8')
    #print(line)
    full_tweet = json.loads(line)
    tweet_text = full_tweet['text']
    print("Tweet Text: " + tweet_text)
    #print ("------------------------------------------")
    #tcp_connection.send(tweet_text + '\n')
    tweet_rdd =  spark.sparkContext.parallelize([tweet_text])
    rdd = rdd.union(tweet_rdd)
  
  return rdd


resp = get_tweets()
rdd = parse_tweets(resp)

print("I read %d tweets" % rdd.count())
#print(rdd.collect())
  
words = rdd.flatMap(lambda line: line.split(" "))
#print(words.collect())

# filter the words to get only hashtags, then map each hashtag to be a pair of (hashtag,1)
hashtags = words.filter(lambda w: '#' in w).map(lambda x: (x, 1))
print("%d hashtags found" % hashtags.count())

hashtags_reduced = hashtags.reduceByKey(lambda a, b: a + b)
print("%d unique hashtags found" % hashtags.count())

