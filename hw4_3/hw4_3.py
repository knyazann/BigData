import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from sklearn.metrics import mean_squared_error


# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_4_3"))

ratings_df = spark.read.csv('./ml-latest/ratings.csv', header=True, inferSchema=True).drop("timestamp")
train_df, test_df = ratings_df.randomSplit([0.8, 0.2])

print(f"\
        Среднее значение рейтинга в обучающем подмножестве: 3.5028150848865907\n \
        RMSE для тестового подмножества, если для всех значений из test предсказывается среднее значение рейтинга: 1.042954721625963\n \
        RMSE для тестового подмножества (коллаб. фильтрация по схожести пользователей): 0.955058567359689\n")


als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

# Настройка параметров модели 
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

# Оценка модели и выбор лучших параметров с использованием кросс-валидации
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

crossval = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=4
)

pipeline = Pipeline(stages=[als])
model = crossval.fit(train_df)
best_model = model.bestModel

# Формирование финальной модели ALS с лучшими параметрами
best_als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    rank=best_model.rank,
    regParam=best_model._java_obj.parent().getRegParam()
)

best_model_fitted = best_als.fit(train_df)
predictions = best_model_fitted.transform(test_df)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"RMSE для тестового подмножества для факторизации матрицы рейтингов c помощью ALS: {rmse}")

# Закрытие Spark
spark.stop()
