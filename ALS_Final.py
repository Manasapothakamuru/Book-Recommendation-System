import numpy as np
import pandas as pd
import cv2
import logging
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
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

#create a spark session
spark=SparkSession \
  .builder \
  .appName("Example") \
  .config("spark.driver.memory", "2g")\
  .getOrCreate()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

#read the data file books_ratings into the dataframe
lines = spark.read.csv("C:\Users\Manasapothakamuru\OneDrive - Cal State Fullerton\Desktop\ADBMS project\Main DataSet\Books_rating.csv")


#selecting the columns that are not required for the process
drop_column_list = ["_c0","_c2","_c4","_c5","_c7","_c8","_c9"]

#create a dataframe with only required columns and column names are user, item(book), ratings for the book
df2 = lines.withColumnRenamed("_c3","user") \
    .withColumnRenamed("_c1","item") \
    .withColumnRenamed("_c6","rating")

df3 = df2.select([column for column in df2.columns if column not in drop_column_list]) 

#create indexers for the string columns, this will convert into indexers for the string datatypes 
indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ['user', 'item','rating']]
for index in indexer:
    index.setHandleInvalid("skip").fit(df3).transform(df3).show

# pipeline is created to input into the machine learning model
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(df3).transform(df3)
transformed.select(['user', 'item','rating','user_index', 'item_index','rating_index'])

#data is split into training and test dataframes with 80% in training set and 20% in test set
(training, test) = transformed.randomSplit([0.8, 0.2])

#Alternating least squares is the algorithm that is used for the classification. 
#indexed columns are sent as parameters to the model 
als=ALS(maxIter=5,
        regParam=0.09,
        rank=25,
        userCol="user_index",
        itemCol="item_index",
        ratingCol="rating_index",
        coldStartStrategy="drop",
        nonnegative=True)

#model is being trained on the training dataset
model=als.fit(training)

#evaluates the test dataset
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating_index",predictionCol="prediction")
predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)

#root mean square error is found for the predicted test dataset
#print("RMSE="+str(rmse))
print("Accuracy="+str(1.96*rmse))
userid=input("enter the user  id:")
test = model.recommendForAllUsers(20 ).filter(col('user')==userid).select("recommendations").collect()
books = []
for item in test[0][0]:        
    books.append(item.item_index)
    
schema = StructType([StructField("item_index",IntegerType(),True)])
books = spark.createDataFrame(books,IntegerType()).toDF("item_index")

# recommends books based on the user id that is entered

transformed\
.select(['item', 'user', 'rating'])\
.filter(col('user')==userid)\
.drop_duplicates(subset=['item'])\
.show()
