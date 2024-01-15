from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from scipy.spatial.distance import squareform, pdist
import numpy as np
from pyspark.sql.functions import col, expr


# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_4_2"))

def compute_prediction_by_user(user_id, item_id, similarity_matrix, train_data):
    try:
        # Пытаемся получить рейтинги для данного фильма
        item_rating = train_data[item_id]
    except KeyError:
        # Если фильм не найден, возвращаем средний рейтинг пользователя
        user_rating = train_data.loc[user_id, :]
        return user_rating[user_rating > 0].mean()

    try:
        user_id = userId_to_index_mapping[user_id]
    except KeyError:
        return item_rating[item_rating > 0].mean()

    user_similarity = similarity_matrix[user_id]
    numerator = np.dot(item_rating, user_similarity)
    denominator = user_similarity[item_rating > 0].sum()

    if denominator == 0 or numerator == 0:
        return item_rating[item_rating > 0].mean()

    return numerator / denominator

# Загружаем данные из файлов
movies_df = spark.read.csv("./ml-latest-small/movies.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("./ml-latest-small/ratings.csv", header=True, inferSchema=True)

# Объединяем данные по movieId
data_df = ratings_df.join(movies_df, "movieId", "inner")

# Разделяем данные на обучающее и тестовое подмножества
train, test = data_df.randomSplit([0.8, 0.2])

# Вычисляем среднее значение рейтинга в обучающем подмножестве
avg_rating_train = train.select(F.mean("rating")).collect()[0][0]

print(f"Среднее значение рейтинга в обучающем подмножестве: {avg_rating_train}")

# Предсказываем среднее значение рейтинга для тестового подмножества
test_predictions = test.withColumn("prediction", F.lit(avg_rating_train))

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(test_predictions)

print(f"RMSE для тестового подмножества, если для всех значений из test предсказывается среднее значение рейтинга: {rmse}")


train_df = train.toPandas()
test_df = test.toPandas()

# Создание матрицы пользователь-фильм
train_user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0.0)

# Вычисление матрицы схожести пользователей
train_user_similarity = 1 - squareform(pdist(train_user_item_matrix, 'cosine'))

# Словари для маппинга
index_to_userId_mapping = dict(enumerate(train_user_item_matrix.index))
userId_to_index_mapping = {userId: index for index, userId in index_to_userId_mapping.items()}

# Тестирование
test_dataset = test_df[['userId', 'movieId']].to_numpy()
test_ratings = test_df['rating'].to_numpy()

predictions = []
for test_sample in test_dataset:
    res = compute_prediction_by_user(test_sample[0], test_sample[1], train_user_similarity, train_user_item_matrix)
    predictions.append(res)

rmse = np.sqrt(np.mean((test_ratings - predictions)** 2))

print(f"RMSE для тестового подмножества (коллаб. фильтрация по схожести пользователей): {rmse}")

# Закрытие Spark
spark.stop()