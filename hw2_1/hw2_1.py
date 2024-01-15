import json
import geopandas as gpd
import plotly.express as plt
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, FloatType, DateType
)
from shapely.geometry import shape, Point

# Инициализация Spark
spark = SparkSession(SparkContext("local", "HW_2_1"))

# Схема данных
data_schema = StructType([
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

trip_data = spark.read.csv("./201902-citibike-tripdata.csv", header=True, schema=data_schema)

with open("./NYC Taxi Zones.geojson") as file:
    nyc_geojson = json.load(file)


frames = {}
for feature in nyc_geojson['features']:
    zone_name = feature['properties']['zone']
    frames[zone_name] = shape(feature['geometry'])

def find_zone(latitude, longitude):
    point = Point(longitude, latitude)
    for zone_name, frame in frames.items():
        if frame.contains(point):
            return zone_name
    return None

zone_udf = udf(find_zone, StringType())

def count_trips_by_station(data, station_id, station_lat, station_long):
    return data.groupBy(station_id, station_lat, station_long).count()

start_count = count_trips_by_station(trip_data, "start station id", "start station latitude", "start station longitude")
end_count = count_trips_by_station(trip_data, "end station id", "end station latitude", "end station longitude")

def match_stations_with_zones(data, station_id, station_lat, station_long, zone_col):
    stations_data = data.select(station_id, station_lat, station_long).distinct()
    stations_data = stations_data.withColumn(zone_col, zone_udf(col(station_lat), col(station_long)))
    return stations_data

start_stations_data = match_stations_with_zones(trip_data, "start station id", "start station latitude", "start station longitude", "start station zone")
end_stations_data = match_stations_with_zones(trip_data, "end station id", "end station latitude", "end station longitude", "end station zone")

def join_trip_counts(stations_data, count_data, station_id, station_lat, station_long):
    return stations_data.join(count_data, (stations_data[station_id] == count_data[station_id]) &
                              (stations_data[station_lat] == count_data[station_lat]) &
                              (stations_data[station_long] == count_data[station_long]), 'right')

start_stations_data = match_stations_with_zones(trip_data, "start station id", "start station latitude", "start station longitude", "start station zone")
end_stations_data = match_stations_with_zones(trip_data, "end station id", "end station latitude", "end station longitude", "end station zone")

start_stations_data = join_trip_counts(start_stations_data, start_count, "start station id", "start station latitude", "start station longitude")
end_stations_data = join_trip_counts(end_stations_data, end_count, "end station id", "end station latitude", "end station longitude")


count_by_zone_start = start_stations_data.groupBy('start station zone').sum('count')
count_by_zone_end = end_stations_data.groupBy('end station zone').sum('count')

def show_trip_counts(count_data, order_col, desc=True):
    count_data.orderBy(order_col, ascending=not desc).show(truncate=False)

print("\n\nДля каждой станции количество начала поездок и количество завершения поездок:")
show_trip_counts(count_by_zone_start, 'sum(count)')

print("\n\nДля каждой станции количество завершения поездок:")
show_trip_counts(count_by_zone_end, 'sum(count)')

def combine_trip_data(start_data, end_data, start_zone_col, end_zone_col, start_col, end_col):
    combined_data = start_data.withColumnRenamed('sum(count)', start_col).withColumnRenamed(start_zone_col, 'station zone')
    combined_data = combined_data.join(end_data.withColumnRenamed('sum(count)', end_col).withColumnRenamed(end_zone_col, 'station zone'),
                                       'station zone', 'fullouter').select('station zone', start_col, end_col)
    combined_data = combined_data.na.fill(0)
    combined_data = combined_data.withColumn('start plus end', combined_data[start_col] + combined_data[end_col]).select('station zone', 'start plus end')
    return combined_data

combined_trip_data = combine_trip_data(count_by_zone_start, count_by_zone_end, 'start station zone', 'end station zone', 'start count', 'end count')

combined_trip_data = combined_trip_data.withColumnRenamed('start station zone', 'station zone')

print("\n\nСтанции по убыванию количества поездок:")
show_trip_counts(combined_trip_data, 'start plus end')

# в виде картограммы (Choropleth)
cartogram = plt.choropleth(combined_trip_data.toPandas(), geojson=nyc_geojson, color='start plus end', locations='station zone', 
                    labels={'start plus end': 'count'}, featureidkey='properties.zone', scope='usa')
cartogram.show()

# Закрытие Spark
spark.stop()