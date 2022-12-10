import numpy as np
import pandas as pd
import cv2
#from matplotlib import pyplot as plt
#from _future_ import print_function
import logging
#import io
#import sys
#import os
#Import svm model
#from sklearn import svm
#import csv
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
#Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics
#import pandas as pd
import numpy as np
import cv2
import glob
from sklearn import svm
from matplotlib import pyplot as plt
import mahotas
import sklearn
import sklearn.preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql.functions import col
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType,StructField,IntegerType


#conf= pyspark.SparkConf().setAppName('SparkApp').setMaster('local')
#sc = SparkContext.getOrCreate()
spark=SparkSession \
  .builder \
  .appName("Example") \
  .config("spark.driver.memory", "16g")\
  .getOrCreate()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.csv("C:\\Users\Manasapothakamuru\OneDrive - Cal State Fullerton\Desktop\ADBMS project\Main DataSet\Books_rating.csv")
drop_column_list = ["_c1","_c2","_c4","_c5","_c7","_c8","_c9"]
df2 = lines.withColumnRenamed("_c3","user") \
    .withColumnRenamed("_c0","item") \
    .withColumnRenamed("_c6","rating")


df3 = df2.select([column for column in df2.columns if column not in drop_column_list]) 
df3.printSchema()
df3.show()
indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ['user', 'item','rating']]
for index in indexer:
    index.setHandleInvalid("skip").fit(df3).transform(df3).show
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(df3).transform(df3)
transformed.select(['user', 'item','rating','user_index', 'item_index','rating_index'])


(training, test) = transformed.randomSplit([0.8, 0.2])


als=ALS(maxIter=5,
        regParam=0.09,
        rank=25,
        userCol="user_index",
        itemCol="item_index",
        ratingCol="rating_index",
        coldStartStrategy="drop",
        nonnegative=True)

model=als.fit(training)
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating_index",predictionCol="prediction")
predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)
print("RMSE="+str(rmse))
print("Accuracy="+str(1.96*rmse))
test = model.recommendForAllUsers(20).filter(col('user_index')==30).select("recommendations").collect()
books = []
for item in test[0][0]:        
    books.append(item.item_index)
    
schema = StructType([StructField("item_index",IntegerType(),True)])
books = spark.createDataFrame(books,IntegerType()).toDF("item_index")


transformed\
.select(['item', 'user', 'rating'])\
.filter(col('user_index')==30)\
.show()

books\
.join(transformed, on = 'item_index', how = 'inner')\
.select(['item', 'rating', 'user'])\
.drop_duplicates(subset=['user'])\
.show()

