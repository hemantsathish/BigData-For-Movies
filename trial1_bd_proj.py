from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.functions import broadcast

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

joined_table = df2.join(df3, ['user_id']).dropDuplicates()
#joined_table.show()

#final_table = joined_table.join(df1, ['movie_id'])
#final_table.show()

mostRated = joined_table.groupby('movie_id').count().orderBy(func.desc("count"))
#mostRated.join(broadcast(df1), ['movie_id']).show()

def details(df):
  df.join(broadcast(df1), ['movie_id']).show()
  
details(mostRated)

spark.stop()
