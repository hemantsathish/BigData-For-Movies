from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.functions import col, explode
from pyspark.sql.functions import broadcast
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType

# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Import the requisite items
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


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
joined_table = joined_table.withColumn("rating_score",joined_table["rating_score"].cast(FloatType()))
joined_table = joined_table.withColumn("user_id",joined_table["user_id"].cast(IntegerType()))
joined_table = joined_table.withColumn("movie_id",joined_table["movie_id"].cast(IntegerType()))
df = joined_table.select(['user_id','movie_id','rating_score'])
#df.show()
'''
df = df.dropna(how='any')
df.where("Value is null").show()
print(df.where(col("rating_score").isNull()))
print(df.where(col("user_id").isNull()))
print(df.where(col("movie_id").isNull()))
'''
(training, test) = df.randomSplit([0.8, 0.2])
training=training.na.fill(0.0)
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="movie_id", ratingCol="rating_score",
          coldStartStrategy="drop")
model = als.fit(training)


test=test.na.fill(0.0)
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating_score",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = df.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
#movies = df.select(als.getItemCol()).distinct().limit(3)
#movieSubSetRecs = model.recommendForItemSubset(movies, 10)
userSubsetRecs = userSubsetRecs.withColumn("rec_exp", explode("recommendations")).select('user_id', col("rec_exp.movie_id"), col("rec_exp.rating"))
userSubsetRecs.limit(10).show()


'''
# Count the total number of ratings in the dataset
numerator = df.select("rating_score").count()

# Count the number of distinct userIds and distinct movieIds
num_users = df.select("user_id").distinct().count()
num_movies = df.select("movie_id").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

# Group data by userId, count ratings
userId_ratings = df.groupBy("user_id").count().orderBy('count', ascending=False)
#userId_ratings.show()

# Group data by userId, count ratings
movieId_ratings = df.groupBy("movie_id").count().orderBy('count', ascending=False)
#movieId_ratings.show()

# Create test and train set
(train, test) = df.randomSplit([0.8, 0.2], seed = 1234)

# Create ALS model
als = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating_score", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

# Confirm that a model called "als" was created
#type(als)


# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()
            #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

           
# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating_score", predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))



# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Confirm cv was built
print(cv)



#Fit cross validator to the 'train' dataset
model = cv.fit(train)

#Extract best model from the cv model above
best_model = model.bestModel


# Print best_model
print(type(best_model))

# Complete the code below to extract the ALS model parameters
print("**Best Model**")

# # Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())

# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())

# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())



# View the predictions
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)


test_predictions.show()



# Generate n Recommendations for all users
nrecommendations = best_model.recommendForAllUsers(10)
nrecommendations.limit(10).show()


nrecommendations = nrecommendations\
    .withColumn("rec_exp", explode("recommendations"))\
    .select('user_id', col("rec_exp.movie_id"), col("rec_exp.rating_score"))

nrecommendations.limit(10).show()
'''




spark.stop()




