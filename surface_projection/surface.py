import itertools

import plotly
import plotly.graph_objects as go
import numpy as np
from numpy import pi

'''
核心思想：任何空间曲面都可以拆成任意形状的基本几何平面的集合：三角、矩形、圆（弧）
三角：三个顶点确定
矩形：中轴线+宽 --> 四个顶点
圆： R + 中心点(x,y)+ 三个角度(https://www.coder.work/article/2412425)

怎么实现平面图形的加和以及投影

投影：
'''


# 注释
class TripleSubSurface(object):
    def __init__(self, points):
        self.vertices = np.array(points)

        self.edges = self.generate_edges()
        self.normal_vector = self.generate_vector(points)

    def generate_edges(self):
        edges = []
        combinations = list(itertools.combinations(self.vertices, 2))
        for combination in combinations:
            edges.append(list(combination))
        return edges

    def generate_vector(self, points):

        vec1 = np.array(points[1]) - np.array(points[0])
        vec2 = np.array(points[2]) - np.array(points[0])

        return np.cross(vec1, vec2)

    def __add__(self, other):
        if np.linalg.matrix_rank(self.vertices - other.vertices) == 1:
            new_vertices = self.vertices
            return np.linalg.matrix_rank((self.vertices - other.vertices))
        else:
            print("Less than 2 vertices or Coincide")

    def _plot(self, debug=False):
        plot_data = [go.Mesh3d(x=self.vertices[:, 0],
                               y=self.vertices[:, 1],
                               z=self.vertices[:, 2],
                               opacity=0.5,
                               color='rgba(244,22,100,0.6)'
                               ),
                     go.Scatter3d(x=self.vertices[:, 0], y=self.vertices[:, 1], z=self.vertices[:, 2])]
        fig = go.Figure(data=plot_data)
        if debug:
            fig.show()
        return plot_data

    def get_plot_data(self):
        return self._plot()


class PolygonSurface(TripleSubSurface):
    def __init__(self, points, length, type):
        vertx = self.generate_vertices(points, length, type)
        # print(vertx)
        # vertx.sort(key=lambda u: (u[0]))
        self.vertices = np.array(vertx)
        # print(vertx)

    def generate_vertices(self, points, length, type):
        vertices = []
        if type == "x":
            vertices.append([points[0][0] + length / 2, points[0][1], points[0][2]])
            vertices.append([points[0][0] - length / 2, points[0][1], points[0][2]])
            vertices.append([points[1][0] + length / 2, points[1][1], points[1][2]])
            vertices.append([points[1][0] - length / 2, points[1][1], points[1][2]])
        if type == "y":
            vertices.append([points[0][0], points[0][1] + length / 2, points[0][2]])
            vertices.append([points[0][0], points[0][1] - length / 2, points[0][2]])
            vertices.append([points[1][0], points[1][1] + length / 2, points[1][2]])
            vertices.append([points[1][0], points[1][1] - length / 2, points[1][2]])
        if type == "z":
            vertices.append([points[0][0], points[0][1], points[0][2] + length / 2])
            vertices.append([points[0][0], points[0][1], points[0][2] - length / 2])
            vertices.append([points[1][0], points[1][1], points[1][2] + length / 2])
            vertices.append([points[1][0], points[1][1], points[1][2] - length / 2])
        return vertices


class CircleSurface(TripleSubSurface):
    def __init__(self, a, b, point, theta, alpha, vec1, vec2):
        self.radius_a = a
        self.radius_b = b
        self.theta = theta
        self.alpha = alpha

        self.vector_parallel = vec1 / np.linalg.norm(vec1)
        self.vector_vertical = vec2 / np.linalg.norm(vec2)

        # print(self.vector_vertical, self.vector_parallel)
        self.center_point = point

        self.v_point, self.p_point = self.generate_circle()

        # print(self.v_point, self.p_point)

    def generate_circle(self):
        normalized_p = self.vector_parallel
        normalized_v = self.vector_vertical

        return [[normalized_v[0] * self.radius_b + self.center_point[0],
                 normalized_v[1] * self.radius_b + self.center_point[1],
                 normalized_v[2] * self.radius_b + self.center_point[2]],
                [normalized_p[0] * self.radius_a + self.center_point[0],
                 normalized_p[1] * self.radius_a + self.center_point[1],
                 normalized_p[2] * self.radius_a + self.center_point[2]]
                ]

    def _plot(self, debug=False):
        gama = np.linspace(self.theta, self.alpha)
        # y = np.linspace(-pi/2, pi/2)
        # x,y = np.meshgrid(x,y)
        # z = np.sin(self.gama)

        # x = x * self.theta
        # x =

        vec1 = self.vector_parallel
        vec2 = self.vector_vertical

        x = self.center_point[0] + self.radius_a * np.cos(gama) * vec1[0] + self.radius_b * np.sin(gama) * vec2[0]
        y = self.center_point[1] + self.radius_a * np.cos(gama) * vec1[1] + self.radius_b * np.sin(gama) * vec2[1]
        z = self.center_point[2] + self.radius_a * np.cos(gama) * vec1[2] + self.radius_b * np.sin(gama) * vec2[2]

        x = np.append(x, values=self.center_point[0])
        y = np.append(y, values=self.center_point[1])
        z = np.append(z, values=self.center_point[2])

        points = np.array([self.v_point, self.center_point, self.p_point])
        plot_data = [go.Mesh3d(x=x.flatten(),
                               y=y.flatten(),
                               z=z.flatten(),
                               opacity=0.5,
                               alphahull=-1,
                               color='rgba(244,22,100,0.6)'
                               ),
                     go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])]
        fig = go.Figure(data=plot_data)

        if debug:
            fig.update_layout(scene_aspectmode='manual',
                              scene_aspectratio=dict(x=1, y=1, z=1))
            fig.show()
        return plot_data

    def get_plot_data(self):
        return self._plot()


class HandSurface(object):
    def __init__(self, num):
        self.finger1 = PolygonSurface(points=[[0, 0, 1], [0, 2, 0]], length=0.2, type='x')
        self.finger2 = PolygonSurface(points=[[0.5, 0, 1], [0.5, 4, 0]], length=0.2, type='x')
        self.finger3 = PolygonSurface(points=[[1, 0, 1], [1, 4, 0]], length=0.2, type='x')
        self.finger4 = PolygonSurface(points=[[1.5, 0, 1], [1.5, 4, 0]], length=0.2, type='x')
        self.finger5 = PolygonSurface(points=[[2, 0, 1], [2, 4, 0]], length=0.2, type='x')

        self.hand_slice1 = TripleSubSurface(points=[[0, 0, 1], [1.1, 0, 1], [0, -4, 0]])
        self.hand_slice2 = TripleSubSurface(points=[[1.1, 0, 1], [2.1, 0, 1], [2.1, -4, 0]])
        self.hand_slice3 = TripleSubSurface(points=[[1.1, -2, 0.5], [0, -4, 0], [2.1, -4, 0]])

    def plot_show(self, debug=False):
        plot_data = self.finger1.get_plot_data()
        # plot_data = np.append(plot_data,self.finger2.get_plot_data())

        fig = go.Figure()
        fig.add_trace(self.finger1.get_plot_data()[0])
        fig.add_trace(self.finger2.get_plot_data()[0])
        fig.add_trace(self.finger3.get_plot_data()[0])
        fig.add_trace(self.finger4.get_plot_data()[0])
        fig.add_trace(self.finger5.get_plot_data()[0])
        fig.add_trace(self.hand_slice1.get_plot_data()[0])
        fig.add_trace(self.hand_slice2.get_plot_data()[0])
        fig.add_trace(self.hand_slice3.get_plot_data()[0])

        if debug:
            # fig.update_layout(scene_aspectmode='auto',
            #                   scene_aspectratio=dict(x=1, y=1, z=1)),
            fig.update_layout(scene=dict(
                xaxis=dict(nticks=4, range=[-10, 10], ),
                yaxis=dict(nticks=4, range=[-10, 10], ),
                zaxis=dict(nticks=4, range=[-10, 10], ), ))
            fig.show()


class SurfaceProjection(object):
    def __init__(self, top_surface, bottom_surface):
        if type(bottom_surface) == TripleSubSurface:
            if type(top_surface) == TripleSubSurface:
                self.points_projection = self.generate_projection()
        print(self.points_projection)

    def generate_projection(self):
        """
        point1 是待投影点
        points 是投影后点
        """
        points = []
        vec1 = [1, 0, 1]
        point1 = [1, 0, 1]
        point2 = [2, 1, 0]
        # print((np.dot(vec1, point2) - np.dot(vec1, point1)) / np.dot(vec1, vec1))
        points = np.dot(((np.dot(vec1, point2) - np.dot(vec1, point1)) / np.dot(vec1, vec1)) , vec1) + point1

        return points


if __name__ == '__main__':
    p = [[1, 2, 3], [7, 7, 7], [4, 2, 6], [5, 7, 6]]
    p2 = [[2, 2, 3], [2, 2, 7], [4, 5, 6]]
    tri = TripleSubSurface(p)

    # tri2 = TripleSubSurface(p2)
    # pol = PolygonSurface(points=[[1, 2, 3], [2, 3, 4]], length=4, type='x')
    # pol._plot()
    # print(tri2 + tri)
    # tri._plot()

    # cir = CircleSurface(a=1, b=1, point=(2, 0, 0), theta=0, alpha=pi, vec1=[1, 0, 1], vec2=[0, 1, 1])
    # cir._plot()

    hand = HandSurface(num=1)
    hand.plot_show(debug=True)

    pro = SurfaceProjection(bottom_surface=tri,top_surface=tri)