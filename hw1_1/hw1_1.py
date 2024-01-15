from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from math import radians, sin, cos, sqrt, atan2

# Инициализация Spark
sc = SparkContext("local", "HW_1_1")
spark = SparkSession(sc)

# Схема данных
schema = StructType([
    StructField("ID", IntegerType()),
    StructField("Name", StringType()),
    StructField("global_id", IntegerType()),
    StructField("IsNetObject", StringType()),
    StructField("OperatingCompany", StringType()),
    StructField("TypeObject", StringType()),
    StructField("AdmArea", StringType()),
    StructField("District", StringType()),
    StructField("Address", StringType()),
    StructField("PublicPhone", StringType()),
    StructField("SeatsCount", IntegerType()),
    StructField("SocialPrivileges", StringType()),
    StructField("Longitude_WGS84", DoubleType()),
    StructField("Latitude_WGS84", DoubleType()),
    StructField("geoData", StringType())
])

# Чтение данных из CSV
data = spark.read.csv("./hw1_1/places.csv", header=False, schema=schema)

# Заданная точка
target_lat, target_lng = 55.751244, 37.618423

# Функция для вычисления расстояния между двумя точками
def calculate_distance(lat1, lng1, lat2, lng2):
    R = 6371.0  # Радиус Земли в км
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Расстояние от заданной точки до каждого заведения общепита и вывод первых 10 результатов
target_distance_rdd = data.rdd.map(lambda row: (
    row["ID"],
    row["Name"],
    calculate_distance(row["Latitude_WGS84"], row["Longitude_WGS84"], target_lat, target_lng)
))

target_distance_top10 = target_distance_rdd.takeOrdered(10, key=lambda x: x[2])
print("Топ-10 заведений, ближайших к заданной точке:")
for item in target_distance_top10:
    print(f"{item[1]}: {item[2]} km")

# Расстояние между всеми заведениями общепита и вывод первых 10 результатов
all_distance_rdd = data.rdd.cartesian(data.rdd).filter(lambda x: x[0]["ID"] < x[1]["ID"]).map(lambda x: (
    (x[0]["Name"], x[1]["Name"]),
    calculate_distance(x[0]["Latitude_WGS84"], x[0]["Longitude_WGS84"], x[1]["Latitude_WGS84"], x[1]["Longitude_WGS84"])
))

all_distance_top10 = all_distance_rdd.takeOrdered(10, key=lambda x: x[1])
print("\nТоп-10 ближаших заведений среди всех:")
for item in all_distance_top10:
    print(f"{item[0]}: {item[1]} km")

# Топ-10 наиболее близких и наиболее отдаленных заведений
closest_and_farthest = all_distance_rdd.takeOrdered(10, key=lambda x: x[1]) + all_distance_rdd.takeOrdered(10, key=lambda x: -x[1])
print("\nТоп-10 наиболее близких и наиболее отдаленных заведений:")
for item in closest_and_farthest:
    print(f"{item[0]}: {item[1]} km")

# Закрытие Spark
spark.stop()