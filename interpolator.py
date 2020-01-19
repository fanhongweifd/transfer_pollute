import psycopg2
from photutils.utils import ShepardIDWInterpolator
import psycopg2.extras
import json
from math import *
import pandas as pd
from scipy.interpolate import griddata
from scipy import *
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


def getDegree(latA, lonA, latB, lonB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))

    # 为了和风场方向相同，正南方向为0
    brng = (brng + 540) % 360
    return brng


def degree2vector(degree):
    degree = (degree + 360) % 360
    if 0 <= degree and degree < 90:
        vector_x = -abs(sin(degree/180.0 * pi))
        vector_y = -abs(cos(degree/180.0 * pi))
    elif 90 <= degree and degree < 180:
        vector_x = -abs(sin(degree/180.0 * pi))
        vector_y = abs(cos(degree / 180.0 * pi))
    elif 180 <= degree and degree < 270:
        vector_x = abs(sin(degree/180.0 * pi))
        vector_y = abs(cos(degree / 180.0 * pi))
    elif 270 <= degree and degree < 360:
        vector_x = abs(sin(degree/180.0 * pi))
        vector_y = -abs(cos(degree / 180.0 * pi))
    sq = hypot(vector_x, vector_y)
    vector_x /= sq
    vector_y /= sq
    return vector_x, vector_y


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
        # contour = plt.contour(xi, yi, interpolated_result,[50, 55, 60, 70, 90, 10000])
        # plt.clabel(contour, fontsize=10)
        # plt.show()
        # xi=np.linspace(min(x),max(x),300)
        # yi=np.linspace(min(y),max(y),300)
    interpolated_results['grid_coordinate'] = (xi, yi)
    return interpolated_results



def get_contour(interpolated_results, interpolated_key , contour_values):
    # 给整个区域的数据，和等高线的值，返回等高线
    # interpolated_results: 插值出来的污染区域
    # interpolated_keys： 污染物
    # contour_values： 等高线的值
    (X, Y) = interpolated_results['grid_coordinate']
    interpolated_result = interpolated_results[interpolated_key]
    contour = plt.contour(X, Y, interpolated_result, contour_values)
    contour_lines = contour.allsegs

    # field_grad = gradient(-interpolated_result)
    # field_grad /= hypot(field_grad[0], field_grad[1])
    ## 这是画向量
    # plt.quiver(X[::2, ::2], Y[::2, ::2], field_grad[0, ::2, ::2], field_grad[1, ::2, ::2],)
    # plt.clabel(contour, fontsize=10)
    # plt.show()
    return contour_lines


def get_gradient(interpolated_results, interpolated_key):
    interpolated_result = interpolated_results[interpolated_key]
    field_grad = gradient(-interpolated_result)
    field_grad /= hypot(field_grad[0], field_grad[1])
    return field_grad


def get_contour_grad(contour_lines):
    # 这个是找到等高线的梯度
    contour_grads = []
    for contour_line in contour_lines:
        contour_grad = []
        for line in contour_line:
            num_points = line.shape[0]
            points = line[1:num_points-1]
            grad_x = [NaN] * (num_points-2)
            grad_y = [NaN] * (num_points-2)
            degree = [NaN] * (num_points-2)
            for i in range(1, num_points-1):
                degree_0 = getDegree(line[i][1], line[i][0], line[i-1][1], line[i-1][0])
                degree_1 = getDegree(line[i][1], line[i][0], line[i + 1][1], line[i + 1][0])
                degree_avg = (degree_0 + degree_1)/2
                dx, dy = degree2vector(degree_avg)
                grad_x[i-1] = dx
                grad_y[i - 1] = dy
                degree[i-1] = degree_avg
            temp = {'points':points, 'grad_x':grad_x, 'grad_y':grad_y, 'degree':degree}
            contour_grad.append(temp)
        contour_grads.append(contour_grad)
    return contour_grads


def check_contour_grad(contour_grads, oridata, key, delta = 0.00001):
    # 纠正所有梯度方向，使其指向衰减的方向
    data = oridata['data'].to_list()
    points = oridata['point'].to_list()
    interpolated_data = pd.DataFrame(data)
    interpolated_data['point'] = points
    interpolated_data.dropna(subset=[key], inplace=True)
    values = interpolated_data[key].dropna().to_list()
    # for contour_grad in contour_grads:
    for contour_i in range(len(contour_grads)):
        for contour_j in range(len(contour_grads[contour_i])):
        # for line in contour_grad:
            coor = np.array(contour_grads[contour_i][contour_j]['points'])
            # 先拟合一下等高线的值
            contour_result = griddata(np.array(interpolated_data['point'].to_list()),
                                           np.array(values), (coor[:,0], coor[:,1]), method='cubic')
            # 再拟合一下梯度方向的值,
            coor_dx = coor[:, 0] + np.array(contour_grads[contour_i][contour_j]['grad_x'])*delta
            coor_dy = coor[:, 1] + np.array(contour_grads[contour_i][contour_j]['grad_y'])*delta
            d_contour_result = griddata(np.array(interpolated_data['point'].to_list()),
                                           np.array(values), (coor_dx, coor_dy), method='cubic')

            wrong_list = np.where((contour_result - d_contour_result)<0)[0]
            for id in wrong_list:
                contour_grads[contour_i][contour_j]['degree'][id] = (contour_grads[contour_i][contour_j]['degree'][id] +180)%360
                dx, dy = degree2vector(contour_grads[contour_i][contour_j]['degree'][id])
                contour_grads[contour_i][contour_j]['grad_x'][id] = dx
                contour_grads[contour_i][contour_j]['grad_y'][id] = dy
    return contour_grads


def check_contour_grad_wind(contour_grads, oridata, key, delta = 20):
    # 去掉和风向不同的梯度
    data = oridata['data'].to_list()
    points = oridata['point'].to_list()
    interpolated_data = pd.DataFrame(data)
    interpolated_data['point'] = points
    interpolated_data.dropna(subset=[key], inplace=True)
    values = interpolated_data[key].dropna().to_list()
    # for contour_grad in contour_grads:
    for contour_i in range(len(contour_grads)):
        for contour_j in range(len(contour_grads[contour_i])):
            # for line in contour_grad:
            coor = np.array(contour_grads[contour_i][contour_j]['points'])
            # 先拟合一下等高线上的风向
            contour_wind = griddata(np.array(interpolated_data['point'].to_list()),
                                      np.array(values), (coor[:, 0], coor[:, 1]), method='cubic')
            degree_diff = abs(np.array(contour_grads[contour_i][contour_j]['degree'])-contour_wind)
            idx = np.where((degree_diff < delta)|(degree_diff > 360-delta))[0]
            if len(idx)>0:
                contour_grads[contour_i][contour_j]['points'] = contour_grads[contour_i][contour_j]['points'][idx,:]
                contour_grads[contour_i][contour_j]['degree'] = np.array(contour_grads[contour_i][contour_j]['degree'])[idx]
                contour_grads[contour_i][contour_j]['grad_x'] = np.array(contour_grads[contour_i][contour_j]['grad_x'])[idx]
                contour_grads[contour_i][contour_j]['grad_y'] = np.array(contour_grads[contour_i][contour_j]['grad_y'])[idx]
            else:
                contour_grads[contour_i][contour_j]['points'] = []
                contour_grads[contour_i][contour_j]['degree'] = []
                contour_grads[contour_i][contour_j]['grad_x'] = []
                contour_grads[contour_i][contour_j]['grad_y'] = []
    return contour_grads




if __name__ == '__main__':
    data_loader = DataLoader()
    station_air_quality_rows = data_loader.load_station_air_quality('2019-05-12 00:00:00')
    station_air_quality_interpolated_results = interpolator(pd.DataFrame(station_air_quality_rows), ["pm10", "aqi"])
    contour_lines = get_contour(station_air_quality_interpolated_results, 'pm10', [50,70,90])
    print(contour_lines)

    contour_grads = get_contour_grad(contour_lines)
    contour_grads = check_contour_grad(contour_grads, pd.DataFrame(station_air_quality_rows), 'pm10', delta=0.00001)

    # # 画梯度
    # interpolated_results = station_air_quality_interpolated_results['pm10']
    # (X, Y) = station_air_quality_interpolated_results['grid_coordinate']
    # contour = plt.contour(X, Y, interpolated_results, [50,70,90])
    #
    # for contour_grad in contour_grads:
    #     for line in contour_grad:
    #         coor = np.array(line['points'])
    #         # plt.quiver(coor[:,0], coor[:,1], line['grad_x'], line['grad_y'])
    #         # intervel = 3
    #         plt.quiver(coor[:, 0][::10], coor[:, 1][::10], line['grad_x'][::10], line['grad_y'][::10], width=0.001,scale=100)
    # plt.show()



    # print(station_air_quality_interpolated_results['pm10'])

    station_weather_rows = data_loader.load_station_hour_weather('2019-05-12 00:00:00')
    contour_grads = check_contour_grad_wind(contour_grads, pd.DataFrame(station_weather_rows), 'wind_degrees')

    # 画梯度
    interpolated_results = station_air_quality_interpolated_results['pm10']
    (X, Y) = station_air_quality_interpolated_results['grid_coordinate']
    contour = plt.contour(X, Y, interpolated_results, [50,70,90])

    for contour_grad in contour_grads:
        for line in contour_grad:
            coor = np.array(line['points'])
            # plt.quiver(coor[:,0], coor[:,1], line['grad_x'], line['grad_y'])
            # intervel = 3
            if len(coor)>0:
                plt.quiver(coor[:, 0][::10], coor[:, 1][::10], line['grad_x'][::10], line['grad_y'][::10], width=0.001,scale=100)
    plt.show()
    # station_weather_interpolated_results = interpolator(pd.DataFrame(station_weather_rows), ["wind_degrees", "wind_speed"])
    # station_weather_interpolated_results = interpolator(pd.DataFrame(station_weather_rows), ["wind_speed"])
    # print(station_weather_interpolated_results['wind_degrees'])


