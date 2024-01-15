import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_1_2"))


# Схема данных
ratings_schema = StructType([
    StructField("item", StringType()),
    StructField("user", StringType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", IntegerType())
])

meta_schema = StructType([
    StructField("asin", StringType()),
    StructField("title", StringType()),
    StructField("feature", StringType()),
    StructField("description", StringType()),
    StructField("price", StringType()),
    StructField("imageURL", StringType()),
    StructField("imageURLHighRes", StringType()),
    StructField("also_buy", ArrayType(elementType=StringType())),
    StructField("also_viewed", ArrayType(elementType=StringType())),
    StructField("salesRank", StringType()),
    StructField("brand", StringType()),
    StructField("tech1", StringType()),
    StructField("tech2", StringType()),

])
ratings_data = spark.read.csv("./hw1_2/AMAZON_FASHION.csv", header=False, schema=ratings_schema)
meta_data = spark.read.json("./hw1_2/meta_AMAZON_FASHION.json", schema=meta_schema)

average_rating = ratings_data.rdd.map(lambda x: x["rating"]).reduce(lambda x, y: x + y) / ratings_data.count()
print(f"Средний рейтинг товаров: {average_rating}")

filepath = "result.csv"
top_products = (
    ratings_data.rdd.map(
        lambda record: (record["item"], record["rating"])
    ).filter(
        lambda score: float(score[1]) < 3.0
    ).join(
        meta_data.rdd.
        map(lambda details: (details["asin"], details["title"])).
        filter(lambda info: info[0] != "" and info[1] != "").
        map(lambda pair: (pair[0], pair[1]))
    ).map(
        lambda combined: (combined[1], combined[1])
    ).reduceByKey(
        lambda first, second: first
    ).map(
        lambda element: element[0]
    ).sortBy(
        lambda item: item[0]
    ).take(10)
)

# Вывод результатов в консоль
print("Топ-10 товаров с наименьшим рейтингом:")
for product in top_products:
    print(product)

# Запись результатов в CSV-файл
with open(filepath, 'w', newline='') as result:
    writer = csv.writer(result)
    writer.writerow(['rating', 'title'])
    writer.writerows(top_products)
    print(f"Результат сохранен в файл {filepath}")

# Закрытие Spark
spark.stop()
