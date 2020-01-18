import psycopg2
from photutils.utils import ShepardIDWInterpolator
import psycopg2.extras
import json
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np


class DataLoader:

    def __init__(self, username="maps-cd-onlyread", password="sadgsdgasp1~!~vokhwog21231", database="maps-cd",
                 port="3433",
                 dialect="postgres", host="pgm-wz9y18974piro53xxo.pg.rds.aliyuncs.com"):
        self.username = username
        self.password = password
        self.database = database
        self.port = port
        self.dialect = dialect
        self.host = host
        self.conn = psycopg2.connect(database=self.database, user=self.username, password=self.password,
                                     host=self.host, port=self.port)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def load_station_hour_weather(self, published_at):
        self.cur.execute("select * from station_hour_weather where published_at = '%s'"%published_at)
        rows = self.cur.fetchall()
        for row in rows:
            row['point'] = (row['coord']['lng'], row['coord']['lat'])
        return rows

    def load_station_air_quality(self, published_at):
        self.cur.execute("select * from station_air_quality where published_at = '%s'"%published_at)
        rows = self.cur.fetchall()
        for row in rows:
            row['point'] = (row['coord']['lng'], row['coord']['lat'])
        return rows



# def interpolator(rows, interpolated_keys):
#     with open("grids.json", "r") as f:
#         grids = json.load(f)
#         # x = set([i[0] for i in grids])
#         # y = set([i[1] for i in grids])
#     interpolated_results = dict()
#
#     data = rows['data'].to_list()
#     points = rows['point'].to_list()
#     interpolated_data = pd.DataFrame(data)
#     interpolated_data['point'] = points
#     for interpolated_key in interpolated_keys:
#         interpolated_data.dropna(subset=[interpolated_key], inplace=True)
#         values = interpolated_data[interpolated_key].dropna().to_list()
#         f = ShepardIDWInterpolator(interpolated_data['point'].to_list(), values)
#         interpolated_result = f(grids, n_neighbors=8)
#         interpolated_results[interpolated_key] = interpolated_result
#     return interpolated_results

def interpolator(rows, interpolated_keys):
    # 给出固定站的数据，插值出整个区域数据
    x = np.linspace(103.078027, 105.160972, 202)
    y = np.linspace(29.903774, 31.457907, 174)
    xi, yi = np.meshgrid(x, y)  # 网格化
    interpolated_results = dict()
    data = rows['data'].to_list()
    points = rows['point'].to_list()
    interpolated_data = pd.DataFrame(data)
    interpolated_data['point'] = points
    for interpolated_key in interpolated_keys:
        interpolated_data.dropna(subset=[interpolated_key], inplace=True)
        values = interpolated_data[interpolated_key].dropna().to_list()
        interpolated_result = griddata(np.array(interpolated_data['point'].to_list()),
                                       np.array(values), (xi, yi), method='cubic')
        interpolated_results[interpolated_key] = interpolated_result
        contour = plt.contour(xi, yi, interpolated_result,[50, 55, 60, 70, 90, 10000])
        plt.clabel(contour, fontsize=10)
        plt.show()
        # xi=np.linspace(min(x),max(x),300)
        # yi=np.linspace(min(y),max(y),300)
    interpolated_results['grid_coordinate'] = (xi, yi)
    return interpolated_results

def get_contour(interpolated_results, interpolated_keys ,contour_value_list):
    # 给整个区域的数据，和等高线的值，返回等高线
    # interpolated_results: 插值出来的每个点
    pass



if __name__ == '__main__':
    data_loader = DataLoader()
    station_air_quality_rows = data_loader.load_station_air_quality('2019-05-12 00:00:00')
    station_air_quality_interpolated_results = interpolator(pd.DataFrame(station_air_quality_rows), ["pm10", "aqi"])
    print(station_air_quality_interpolated_results['pm10'])


    station_weather_rows = data_loader.load_station_hour_weather('2019-05-12 00:00:00')
    station_weather_interpolated_results = interpolator(pd.DataFrame(station_weather_rows), ["wind_degrees", "wind_speed"])
    print(station_weather_interpolated_results['wind_degrees'])


