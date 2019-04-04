# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:38:29 2019

@author: hp507
"""

def calc_sentiment(rdd):
    import stanfordnlp
    stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
    nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc.sentences[0].print_dependencies()
    

calc_sentiment(None)