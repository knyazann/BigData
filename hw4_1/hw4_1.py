from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.sql.window import Window
from pyspark.sql import functions as F


# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_4_1"))

# Загружаем данные из файлов CSV
movies_df = spark.read.csv('./ml-latest/movies.csv', header=True)
ratings_df = spark.read.csv('./ml-latest/ratings.csv', header=True)

# Задание 1: Выводим данные, сопоставляющие жанры и количество фильмов
movies_df = movies_df.withColumn("genres", F.split(col("genres"), "\\|"))
genres_df = movies_df.select("movieId", explode("genres").alias("genre"))
genres_count = genres_df.groupBy("genre").count().orderBy("count", ascending=False)
print("Количество фильмов по всем жанрам:")
genres_count.show()

print('VARIANT 1')
selected_genres = ['Animation', 'Romance', 'Documentary']
genres_filtered = genres_df.filter(col("genre").isin(selected_genres))
genre_counts = genres_filtered.groupBy("genre").count().orderBy("count", ascending=False)

movies_ratings = movies_df.join(ratings_df, on="movieId")
genres_df = movies_ratings.select("movieId", explode("genres").alias("genre"), "title", "rating")
genres_filtered = genres_df.filter(col("genre").isin(selected_genres))

# Задание 2: Выводим первые 10 фильмов с наибольшим количеством рейтингов для каждого жанра
genre_ratings_info = genres_filtered.groupBy(
    "genre", "movieId", "title"
).agg(
    F.mean("rating").alias("average_rating"),
    F.count("rating").alias("num_ratings")
).filter(
    col("num_ratings") > 10
)

print("Топ-10 фильмов с наибольшим количеством рейтингов для каждого жанра:")
for genre in selected_genres:
    # Использование оконной функции для выбора топ-10
    window = Window.partitionBy("genre").orderBy(col("num_ratings").desc())
    top_films = genre_ratings_info.filter(
        col("genre") == genre
    ).withColumn(
        "rank", F.rank().over(window)
    ).orderBy("rank")

    print(f"\nдля жанра {genre}:")
    top_films.select("movieId", "title", "num_ratings").show(truncate=False, n=10)

# Задание 3: Выводим первые 10 фильмов с наименьшим количеством рейтингов (но больше 10) для каждого жанра
print("Первые 10 фильмов с наименьшим количеством рейтингов для каждого жанра:")
for genre in selected_genres:
    # Использование оконной функции для выбора топ-10
    window = Window.partitionBy("genre").orderBy(col("num_ratings").asc())
    top_films = genre_ratings_info.filter(
        col("genre") == genre
    ).withColumn(
        "rank", F.rank().over(window)
    ).orderBy("rank")

    print(f"\nдля жанра {genre}:")
    top_films.select("movieId", "title", "num_ratings").show(truncate=False, n=10)

# Задание 4: Выводим первые 10 фильмов с наибольшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра
print("-----------------------------------------------------")
print("Топ-10 фильмов с наибольшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра:")
for genre in selected_genres:
    window = Window.partitionBy("genre").orderBy(col("average_rating").desc())
    top_films = genre_ratings_info.filter(
        col("genre") == genre
    ).withColumn(
        "rank", F.rank().over(window)
    ).orderBy("rank")

    print(f"\nдля жанра {genre}:")
    top_films.select("movieId", "title", "average_rating").show(truncate=False, n=10)

# Задание 5: Выводим первые 10 фильмов с наименьшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра
print("Топ-10 фильмов с наименьшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра:")
for genre in selected_genres:
    window = Window.partitionBy("genre").orderBy(col("average_rating").asc())
    top_films = genre_ratings_info.filter(
        col("genre") == genre
    ).withColumn(
        "rank", F.rank().over(window)
    ).orderBy("rank")

    print(f"\nдля жанра {genre}:")
    top_films.select("movieId", "title", "average_rating").show(truncate=False, n=10)


# Закрытие Spark
spark.stop()
