from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType
import folium
from folium.plugins import HeatMapWithTime
import pandas as pd

# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_2_3"))

# Схема данных
trip_schema = StructType([
    StructField("tripduration", IntegerType()),
    StructField("starttime", TimestampType()),
    StructField("stoptime", TimestampType()),
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

# Чтение данных из CSV
trip_data = spark.read.csv("./201902-citibike-tripdata.csv", header=True, schema=trip_schema)

# Подсчет количества поездок с группировкой по станции и дню
daily_start_station_counts = trip_data.groupBy('start station id', F.dayofmonth('starttime').alias('day')).agg({"starttime": "count"}).orderBy('day')
daily_end_station_counts  = trip_data.groupBy('end station id', F.dayofmonth('stoptime').alias('day')).agg({"stoptime": "count"}).orderBy('day')

# Расчет среднего количества начал поездок и окончаний поездок по станции за день
average_daily_start  = daily_start_station_counts.groupBy('start station id').agg({'count(starttime)': 'mean'}).withColumnRenamed('avg(count(starttime))', 'mean start count')
average_daily_end  = daily_end_station_counts.groupBy('end station id').agg({'count(stoptime)': 'mean'}).withColumnRenamed('avg(count(stoptime))', 'mean end count')

daily_average_trips  = average_daily_start.join(average_daily_end, (average_daily_start['start station id'] == average_daily_end['end station id']), 'inner')
daily_average_trips  = daily_average_trips.select('start station id', 'mean start count', 'mean end count').withColumnRenamed('start station id', 'station id')

print("Среднее количество начала поездок и количество завершения поездок В ДЕНЬ:")
daily_average_trips.show()

# Добавление колонок для часа начала и окончания поездки
trip_data = trip_data.withColumn("starthour", F.hour("starttime"))
trip_data = trip_data.withColumn("endhour", F.hour("stoptime"))

# Определение времени дня для начала и окончания поездки
def get_time_of_day(hour):
    return F.when((hour >= 6) & (hour <= 11), "утро")\
            .when((hour >= 12) & (hour <= 17), "день")\
            .when((hour >= 18) & (hour <= 23), "вечер")\
            .otherwise("ночь")

trip_data = trip_data.withColumn("start time of day", get_time_of_day(F.col("starthour")))
trip_data = trip_data.withColumn("end time of day", get_time_of_day(F.col("endhour")))

# Подсчет количества поездок по дням и станциям
trip_data = trip_data.withColumn('startday', F.dayofmonth('starttime'))
daily_starts = (
    trip_data.groupBy('start time of day', 'start station id', 'startday')
    .count()
    .orderBy('startday', 'start station id')
)

# Вычисление среднего количества начальных поездок
average_starts = (
    daily_starts.groupBy('start station id', 'start time of day')
    .agg(F.mean("count").alias("mean starts"))
)

# Получение координат станций
start_stations_coordinates = (
    trip_data.select('start station id', 'start station latitude', 'start station longitude')
    .distinct()
)

# Объединение данных со средними значениями и координатами
average_starts = average_starts.join(start_stations_coordinates, 'start station id', 'left')

# Вывод результата
print("Среднее количество начала поездок УТРОМ, ДНЕМ, ВЕЧЕРОМ, НОЧЬЮ")
average_starts.orderBy('start station id', 'start time of day').show()

daily_ends = trip_data.withColumn('endday', F.dayofmonth(trip_data['stoptime'])).groupBy('end time of day', 'end station id', 'endday').count().orderBy('endday','end station id')
average_ends = daily_ends.groupBy('end station id', 'end time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean ends')

end_stations_coordinates = trip_data.select('end station id', 'end station latitude', 'end station longitude').distinct()

average_ends = average_ends.join(end_stations_coordinates, ['end station id'], 'left')
print("Среднее количество завершения поездок УТРОМ, ДНЕМ, ВЕЧЕРОМ, НОЧЬЮ")
average_ends.orderBy('end station id', 'end time of day').show()

# Преобразование данных с добавлением дня недели
trip_data = trip_data.withColumn("day of week", F.dayofweek("starttime"))

# Фильтрация данных по средам и воскресеньям
data_wed = trip_data.where(trip_data["day of week"] == 4)
data_sun = trip_data.where(trip_data["day of week"] == 1)

# Расчет количества поездок в среду с группировкой по дню начала
wed_start_grouped = data_wed.withColumn("startday", F.dayofmonth("starttime")) \
    .groupBy("start time of day", "start station id", "startday").count() \
    .orderBy("startday", "start station id")

# Расчет среднего количества начал поездок по станциям и времени дня
wed_avg_starts = wed_start_grouped.groupBy("start station id", "start time of day") \
    .agg(F.mean("count").alias("mean starts"))

# Расчет количества поездок в среду с группировкой по дню окончания
wed_end_grouped = data_wed.withColumn("endday", F.dayofmonth("stoptime")) \
    .groupBy("end time of day", "end station id", "endday").count() \
    .orderBy("endday", "end station id")

# Расчет среднего количества окончаний поездок по станциям и времени дня
wed_avg_ends = wed_end_grouped.groupBy("end station id", "end time of day") \
    .agg(F.mean("count").alias("mean ends"))

# Объединение данных по началу и окончанию поездок
wed_combined = wed_avg_starts.join(wed_avg_ends, 
                                   (wed_avg_starts["start station id"] == wed_avg_ends["end station id"]) & 
                                   (wed_avg_starts["start time of day"] == wed_avg_ends["end time of day"]), 
                                   "inner")

# Формирование итогового DataFrame
wed_final_df = wed_combined.select("start station id", "start time of day", "mean starts", "mean ends") \
    .withColumnRenamed("start station id", "station id") \
    .withColumnRenamed("start time of day", "time of day") \
    .orderBy("station id", "time of day")

# Вывод результатов
print("Среднее количество начала и окончания поездок по временным диапазонам (СРЕДА):")
wed_final_df.show()

# Расчет количества поездок в воскресенье с группировкой по дню начала
sun_start_grouped = data_sun.withColumn("startday", F.dayofmonth("starttime")) \
    .groupBy("start time of day", "start station id", "startday").count() \
    .orderBy("startday", "start station id")

# Расчет среднего количества начал поездок по станциям и времени дня
sun_avg_starts = sun_start_grouped.groupBy("start station id", "start time of day") \
    .agg(F.mean("count").alias("mean starts"))

# Расчет количества поездок в воскресенье с группировкой по дню окончания
sun_end_grouped = data_sun.withColumn("endday", F.dayofmonth("stoptime")) \
    .groupBy("end time of day", "end station id", "endday").count() \
    .orderBy("endday", "end station id")

# Расчет среднего количества окончаний поездок по станциям и времени дня
sun_avg_ends = sun_end_grouped.groupBy("end station id", "end time of day") \
    .agg(F.mean("count").alias("mean ends"))

# Объединение данных по началу и окончанию поездок
sun_combined = sun_avg_starts.join(sun_avg_ends, 
                                   (sun_avg_starts["start station id"] == sun_avg_ends["end station id"]) & 
                                   (sun_avg_starts["start time of day"] == sun_avg_ends["end time of day"]), 
                                   "inner")

# Формирование итогового DataFrame
sun_final_df = sun_combined.select("start station id", "start time of day", "mean starts", "mean ends") \
    .withColumnRenamed("start station id", "station id") \
    .withColumnRenamed("start time of day", "time of day") \
    .orderBy("station id", "time of day")

# Вывод результатов
print("Среднее количество начала и окончания поездок по временным диапазонам (ВОСКРЕСЕНЬЕ):")
sun_final_df.show()

# Convert Spark DataFrame to Pandas DataFrame
avg_starts_pandas = average_starts.toPandas()

# Create a Folium map centered around the mean latitude and longitude of start stations
heatmap = folium.Map(location=[avg_starts_pandas['start station latitude'].mean(), avg_starts_pandas['start station longitude'].mean()], zoom_start=12)

# Extract unique time of day values
unique_times = avg_starts_pandas['start time of day'].unique()

# Prepare data for HeatMapWithTime
map_data = []
for time in unique_times:
    map_data.append(list(zip(
        avg_starts_pandas[avg_starts_pandas['start time of day'] == time]['start station latitude'],
        avg_starts_pandas[avg_starts_pandas['start time of day'] == time]['start station longitude'],
        avg_starts_pandas[avg_starts_pandas['start time of day'] == time]['mean starts']
    )))

HeatMapWithTime(map_data, index=list(unique_times)).add_to(heatmap)

# Save the map as an HTML file
heatmap.save('./hw2_3/heatmap.html')

# Закрытие Spark
spark.stop()