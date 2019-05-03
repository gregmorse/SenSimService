import cherrypy
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re

class SpacyModel:
    def run(self, s1, s2):        
        doc1 = self.nlp(s1)
        doc2 = self.nlp(s2)
        return doc1.similarity(doc2)

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

class MyWebService(object):

   @cherrypy.expose
   def callspacy(self,string1="",string2=""):      
      output = self.mspacy.run(string1,string2)
      return str(output)

   @cherrypy.expose
   def callUSEDAN(self,data):
       np.set_printoptions(threshold=np.nan)
       data = data.split(',')
       with tf.Session(graph = self.graphDAN) as session:     
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          return str(session.run(self.embedDAN(data)))

   @cherrypy.expose
   def callUSETrans(self,data):
       np.set_printoptions(threshold=np.nan)
       data = data.split(',')
       with tf.Session(graph = self.graphTrans) as session:     
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          return str(session.run(self.embedTrans(data)))


   def __init__(self):
       #spacy
       self.mspacy = SpacyModel()
       
       #USE-DAN         
       self.graphDAN = tf.Graph()
       with tf.Session(graph = self.graphDAN) as session:
        self.embedDAN = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

       #USE-Trans
       self.graphTrans = tf.Graph()
       with tf.Session(graph = self.graphTrans) as session:
        self.embedTrans = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")


if __name__ == '__main__':
   #config = {'server.socket_host': '0.0.0.0'}
   #cherrypy.config.update(config)
   cherrypy.quickstart(MyWebService())