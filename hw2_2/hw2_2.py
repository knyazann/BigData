from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, max, mean, stddev, percentile_approx
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType
from math import radians, sin, cos, sqrt, atan2

# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_2_2"))

# Схема данных
trip_schema = StructType([
    StructField("tripduration", IntegerType()),
    StructField("starttime", DateType()),
    StructField("stoptime", DateType()),
    StructField("start station id", IntegerType()),
    StructField("start station name", StringType()),
    StructField("start station latitude", FloatType()),
    StructField("start station longitude", FloatType()),
    StructField("end station id", IntegerType()),
    StructField("end station name", StringType()),
    StructField("end station latitude", FloatType()),
    StructField("end station longitude", FloatType()),
    StructField("bikeid", IntegerType()),
    StructField("usertype", StringType()),
    StructField("birth year", IntegerType()),
    StructField("gender", IntegerType())
])

# Функция для вычисления расстояния между двумя точками
def calculate_distance(lat1, lng1, lat2, lng2):
    R = 6371.0  # Радиус Земли в км
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000
    return distance

# Чтение данных
trip_data = spark.read.csv("./201902-citibike-tripdata.csv", header=True, schema=trip_schema)

distance_udf = udf(calculate_distance, FloatType())

trip_data_filtered = trip_data.filter(trip_data['start station id'] != trip_data['end station id']).withColumn('distance', distance_udf('start station latitude', 'start station longitude', 'end station latitude', 'end station longitude'))
max_distance = trip_data_filtered.select(max(trip_data_filtered.distance)).collect()[0][0]
average_distance = trip_data_filtered.select(mean(trip_data_filtered.distance)).collect()[0][0]
stddev_distance = trip_data_filtered.select(stddev(trip_data_filtered.distance)).collect()[0][0]
median_distance = trip_data_filtered.select(percentile_approx(trip_data_filtered.distance, 0.5)).collect()[0][0]

print(f"Максимальная дистанция: {max_distance}")
print(f"Средняя дистанция: {average_distance}")
print(f"Стандартное отклонение: {stddev_distance}")
print(f"Медиана: {median_distance}")

# Закрытие Spark
spark.stop()