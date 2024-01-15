import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col


# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_1_3"))

# Схема данных
ratings_schema = StructType([
    StructField("userId", IntegerType()),
    StructField("movieId", IntegerType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", IntegerType())
])

movies_schema = StructType([
    StructField("movieId", IntegerType()),
    StructField("title", StringType()),
    StructField("genres", StringType())
])

# косинусное сходство
def cosine_similarity(ratings1, ratings2):
    dot_product = np.dot(ratings1, ratings2)
    norm_product = np.linalg.norm(ratings1) * np.linalg.norm(ratings2)
    similarity = dot_product / norm_product

    return similarity

TARGET_MOVIE_ID = 589 # заданный фильм
ratings_data = spark.read.csv("./ml-latest-small/ratings.csv", header=True, schema=ratings_schema)
movies_data = spark.read.csv("./ml-latest-small/movies.csv", header=True, schema=movies_schema)


movies = ratings_data.groupBy("movieId") \
    .agg({"rating": "avg"}) \
    .rdd.map(lambda row: (row["movieId"], np.array([row["avg(rating)"]])))


target_movie = (
    ratings_data.filter(ratings_data.movieId == TARGET_MOVIE_ID)
    .rdd.map(lambda row: np.array([row["rating"]]))
    .first()[0])

similarities_rdd = movies.map(
    lambda movie: (movie[0], float(cosine_similarity(movie[1], target_movie)))
)


transformed_rdd = similarities_rdd.toDF(["ID", "Score"]) \
    .orderBy("Score", ascending=False) \
    .join(movies_data, col("ID") == movies_data.movieId) \
    .select(col("ID").alias("movieId"), "title", col("Score").alias("Similarity").cast("double"))

transformed_rdd.show(10, truncate=False)

spark.stop()
