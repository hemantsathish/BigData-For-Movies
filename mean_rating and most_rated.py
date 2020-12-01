from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
import matplotlib
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

file1 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_movie_data.csv")
file2 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_data.csv")
file3 = spark.read.option("header", "true").option("inferSchema", "true").csv("mubi_ratings_user_data.csv")

# Columns to drop
drop_cols1=['movie_url','movie_title_language','movie_image_url','director_url']
drop_cols2=['rating_url','rating_timestamp_utc','critic','critic_likes','critic_comments','user_trialist','user_subscriber','user_eligible_for_trial','user_has_payment_method']
drop_cols3=['rating_date_utc','user_avatar_image_url','user_cover_image_url','user_eligible_for_trial','user_has_payment_method']

df1 = file1.drop(*drop_cols1)
#df1.show()
df2 = file2.drop(*drop_cols2)
#df2.show()
df3 = file3.drop(*drop_cols3)
#df3.show()

#INTERMEDIATE TABLE
print("\n----INTERMEDIATE TABLE----\n")
joined_table = df2.join(df3, ['user_id']).dropDuplicates()
joined_table = joined_table.withColumn("rating_score",joined_table["rating_score"].cast(IntegerType()))
joined_table.show()

#FINAL TABLE
print("\n----FINAL TABLE----\n")
final_table = joined_table.join(df1, ['movie_id'])
final_table.show()

#MOST RATED MOVIE
print("\n----MOST RATED MOVIES----\n")
most_rated = final_table.groupby('movie_id').count().orderBy(func.desc("count"))
most_rated = most_rated.join(df1, ['movie_id'])
most_rated.orderBy("count", ascending=False).show()

#MEAN MOVIE RATING
print("\n----HIGHEST RATED MOVIES----\n")
df2 = df2.withColumn('rating_score', df2['rating_score'].cast(IntegerType()))
df2 = df2.withColumn('movie_id', df2['movie_id'].cast(IntegerType()))
df1 = df1.withColumn('movie_release_year', df1['movie_release_year'].cast(IntegerType()))
df2.na.drop("any")
mean_rating = df2.groupBy('movie_id').agg(func.mean('rating_score'), func.count('rating_score'))
mean_rating = mean_rating.orderBy('avg(rating_score)', ascending=False)
mean_rating = mean_rating.where(mean_rating['count(rating_score)'] > 100)
mean_rating = mean_rating.join(df1, ['movie_id'])
mean_rating.orderBy("avg(rating_score)", ascending=False).show()

#POPULAR DIRECTORS(MOVIE COUNT)
print("\n----DIRECTORS WITH MOST RELEASES----\n")
pop_dir=df1.groupBy("director_name").count().orderBy(func.desc("count"))
pop_dir.na.drop("any")
pop_dir.orderBy(func.desc("count")).show()

#YEAR WITH MOST RELEASES
print("----YEAR WITH MOST RELEASES-----")
max_rel=df1.groupBy("movie_release_year").count().orderBy(func.desc("count"))
max_rel.orderBy(func.desc("count")).show()

#YEAR WITH BEST MOVIES
print("----YEAR WITH BEST MOVIES-----")
best_year = mean_rating.groupBy('movie_release_year').agg(func.mean('avg(rating_score)'), func.count('movie_release_year'))
best_year = best_year.where(best_year['count(movie_release_year)'] > 100)
best_year.orderBy("avg(avg(rating_score))", ascending=False).show()

spark.stop()
