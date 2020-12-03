#importing the required libraries

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.functions import col, explode
from pyspark.sql.functions import broadcast
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline 

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

#importing datasets
file1 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_movie_data.csv")
file2 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_data.csv")
file3 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_user_data.csv")

#Columns to drop
drop_cols1=['movie_url','movie_title_language','movie_image_url','director_url','movie_popularity']
drop_cols2=['rating_url','rating_timestamp_utc','critic','critic_likes','critic_comments','user_trialist','user_subscriber','user_eligible_for_trial','user_has_payment_method']
drop_cols3=['rating_date_utc','user_avatar_image_url','user_cover_image_url','user_eligible_for_trial','user_has_payment_method']

df1 = file1.drop(*drop_cols1)
#df1.show()
df2 = file2.drop(*drop_cols2)
#df2.show()
df3 = file3.drop(*drop_cols3)
#df3.show()
df2 = df2.withColumn('rating_score', df2['rating_score'].cast(IntegerType()))
df2 = df2.withColumn('movie_id', df2['movie_id'].cast(IntegerType()))
df1 = df1.withColumn('movie_release_year', df1['movie_release_year'].cast(IntegerType()))
df2.na.drop("any")

#dropping duplicates
df1 = df1.dropDuplicates()
df2 = df2.dropDuplicates()
df3 = df3.dropDuplicates()

joined_table = df2.join(df3, ['user_id']).dropDuplicates()
#joined_table = joined_table.withColumn("rating_score",joined_table["rating_score"].cast(IntegerType()))
joined_table = joined_table.withColumn("user_id",joined_table["user_id"].cast(IntegerType()))
#joined_table = joined_table.withColumn("movie_id",joined_table["movie_id"].cast(IntegerType()))

final_table = joined_table.join(df1, ['movie_id'])
#final_table.show()

#broadcasting the dataframe with movie details
def details(df):
	df.join(broadcast(df1), ['movie_id']).show(truncate=False)
  
#function with query for most rated movie
def mostRatedMovie():
	#mostRated = joined_table.groupby('movie_id').count().orderBy(func.desc("count"))
	#mostRated.show()
	#mostRated = details(mostRated)
	#mostRated.orderBy(func.desc("count"))
	#mostRated.join(broadcast(df1), ['movie_id']).show()
	mostRated = final_table.groupby('movie_id').count().orderBy(func.desc("count"))
	mostRated = mostRated.join(df1, ['movie_id'])
	mostRated = mostRated.orderBy("count", ascending=False)
	return mostRated

def meanRating():
	mean_rating = df2.groupBy('movie_id').agg(func.mean('rating_score'), func.count('rating_score'))
	mean_rating = mean_rating.orderBy('avg(rating_score)', ascending=False)
	mean_rating = mean_rating.where(mean_rating['count(rating_score)'] > 100)
	mean_rating = mean_rating.join(df1, ['movie_id'])
	return mean_rating


def meanRating():
	mean_rating = df2.groupBy('movie_id').agg(func.mean('rating_score').alias('rating_score'), func.count('rating_score').alias('count'))
	mean_rating = mean_rating.orderBy('rating_score', ascending=False)
	mean_rating = mean_rating.where(mean_rating['count'] > 100)
	mean_rating = mean_rating.join(df1, ['movie_id'])
	#mean_rating = details(mean_rating)
	return mean_rating
	
def highestRated():
	mean = meanRating()
	highestRated = mean.orderBy("rating_score", ascending=False)
	return highestRated

def popularDirectors():
	pop_dir=df1.groupBy("director_name").count().orderBy(func.desc("count"))
	pop_dir = pop_dir.na.drop("any")
	pop_dir = pop_dir.filter(pop_dir.director_name != '(Unknown)')
	pop_dir = pop_dir.orderBy(func.desc("count"))
	return pop_dir

def highestMovies_year():
	max_rel=df1.groupBy("movie_release_year").count().orderBy(func.desc("count"))
	max_rel = max_rel.orderBy(func.desc("count"))
	return max_rel

def bestYear():
	mean = meanRating()
	best_year = mean.groupBy('movie_release_year').agg(func.mean('rating_score').alias('rating_score'), func.count('movie_release_year').alias('count'))
	best_year = best_year.where(best_year['count'] > 100)
	best_year = best_year.orderBy("rating_score", ascending=False)
	return best_year

a = mostRatedMovie()
a.show(truncate=False)
print("^^^^^ Most Rated Movie ^^^^^\n")


b = highestRated()
b.show()
print("^^^^^ Highest Rated Movie ^^^^^\n")


c = popularDirectors()
c.show(truncate=False)
print("^^^^^ Most Popular Directors ^^^^^\n")

d = highestMovies_year()
d.show(truncate=False)
print("^^^^^ Year with the highest number of movies ^^^^^\n")

e = bestYear()
e.show(truncate=False)
print("^^^^^ Year with the best movies ^^^^^\n")


#Recommender System using ALS (Collaborative Filtering)

#train test split
df_rec = joined_table.select(['user_id','movie_id','rating_score'])
(training, test) = df_rec.randomSplit([0.8, 0.2])
training=training.na.fill(0.0)

# Build the recommendation model using ALS on the training data
# Set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

als = ALS(maxIter=10, regParam=0.05, rank=20, userCol="user_id", itemCol="movie_id", ratingCol="rating_score",
          coldStartStrategy="drop")
          
#tried parameters
#10,0.01,10
#5,0.01,10
#10,0.05,20

model = als.fit(training)

test=test.na.fill(0.0)
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating_score",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Square Error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)

# Generate top 10 movie recommendations for a specified set of users

users = df_rec.select(als.getUserCol()).distinct().limit(3)
users_subset = model.recommendForUserSubset(users, 10)


#users_subset = users_subset.withColumn("rec_exp", explode("recommendations")).select('user_id', col("rec_exp.movie_id"), col("rec_exp.rating"))
#users_subset.limit(10).show()
#userRecs = userRecs.withColumn("rec_exp", explode("recommendations")).select('user_id', col("rec_exp.movie_id"))
users_subset = users_subset.withColumn("rec_exp", explode("recommendations")).select('user_id', col("rec_exp.movie_id"))

details(users_subset)
#details(userRecs)
print("^^^^^ Recommendations ^^^^^")

spark.stop()
