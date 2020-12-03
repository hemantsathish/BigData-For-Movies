from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.functions import broadcast
from pyspark.sql.types import IntegerType
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline 


# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

#lines = spark.sparkContext.textFile("fakefriends.csv")
file1 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_movie_data.csv")
file2 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_data.csv")
file3 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_user_data.csv")

#Columns to drop
drop_cols1=['movie_url','movie_title_language','movie_image_url','director_url']
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
joined_table = joined_table.withColumn("rating_score",joined_table["rating_score"].cast(IntegerType()))

#joined_table.show()

final_table = joined_table.join(df1, ['movie_id'])
#final_table.show()

def mostRatedMovie():
	mostRated = final_table.groupby('movie_id').count().orderBy(func.desc("count"))
	mostRated = mostRated.join(df1, ['movie_id'])
	mostRated = mostRated.orderBy("count", ascending=False)
	return mostRated

#graph for top 5 most rated movies
a= mostRatedMovie()
group = a.orderBy("count", ascending=False).select(['movie_title','count']).limit(5)
x=group.toPandas()['movie_title'].values.tolist()
y=group.toPandas()['count'].values.tolist()
plt.bar(x,y)
plt.show()

def details(df):
	df.join(broadcast(df1), ['movie_id']).show()

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

#graph for top 5 highest rated movies
to_plot = highestRated()
to_plot = to_plot.limit(5)
x=to_plot.toPandas()['movie_title'].values.tolist()
y=to_plot.toPandas()['rating_score'].values.tolist()
plt.scatter(x,y)
plt.show()

spark.stop()
