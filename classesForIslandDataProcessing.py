import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib.widgets import TextBox
from sklearn.cluster import DBSCAN,OPTICS


class Interactive3DScatter:
    def __init__(self, df):
        self.x = df['lon_ph'].values
        self.y = df['lat_ph'].values
        self.z = df['h_ph'].values
        self.value1 = None
        self.value2 = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Enter values in boxes for Ocean (Blue) and Seperation (Red)')
        self.init_plot()

    def init_plot(self):
        self.scatter = self.ax.scatter(self.x, self.y, self.z, c='grey')
        self.axbox1 = plt.axes([0.1, 0.01, 0.35, 0.05])
        self.text_box1 = TextBox(self.axbox1, 'Value 1', initial="-91")
        self.axbox2 = plt.axes([0.55, 0.01, 0.35, 0.05])
        self.text_box2 = TextBox(self.axbox2, 'Value 2', initial="-93.5")
        self.text_box1.on_submit(self.update)
        self.text_box2.on_submit(self.update)

    def update(self, _):
        try:
            self.value1 = float(self.text_box1.text)
            self.value2 = float(self.text_box2.text)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return
        distances1 = np.abs(self.z - self.value1)
        distances2 = np.abs(self.z - self.value2)
        colors = np.where(distances1 < distances2, 'blue', 'red')
        print(f"Value 1: {self.value1}, Value 2: {self.value2}")
        # print(f"Colors: {colors}")
        self.scatter.set_color(colors)
        self.fig.canvas.draw_idle()

    def get_final_params(self):
        return self.value1, self.value2
class Interactive2DScatter:
    def __init__(self, df):
        self.x = df['distance'].values
        self.y = df['MSL'].values
        # self.z = df['h_ph'].values
        self.value1 = None
        self.value2 = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        # self.ax.set_zlabel('Z')
        self.ax.set_title('Enter values in boxes for Ocean (Blue) and Seperation (Red)')
        self.init_plot()

    def init_plot(self):
        self.scatter = self.ax.scatter(self.x, self.y, c='grey')
        self.axbox1 = plt.axes([0.1, 0.01, 0.35, 0.05])
        self.text_box1 = TextBox(self.axbox1, 'Value 1', initial="0")
        self.axbox2 = plt.axes([0.55, 0.01, 0.35, 0.05])
        self.text_box2 = TextBox(self.axbox2, 'Value 2', initial="-2")
        self.text_box1.on_submit(self.update)
        self.text_box2.on_submit(self.update)

    def update(self, _):
        try:
            self.value1 = float(self.text_box1.text)
            self.value2 = float(self.text_box2.text)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return
        distances1 = np.abs(self.y - self.value1)
        distances2 = np.abs(self.y - self.value2)
        colors = np.where(distances1 < distances2, 'blue', 'red')
        print(f"Value 1: {self.value1}, Value 2: {self.value2}")
        # print(f"Colors: {colors}")
        self.scatter.set_color(colors)
        self.fig.canvas.draw_idle()

    def get_final_params(self):
        return self.value1, self.value2
class InteractiveDBSCAN3D:
    def __init__(self, df):
        self.x = df['lon_ph'].values
        self.y = df['lat_ph'].values
        self.z = df['h_ph'].values
        self.eps = None
        self.min_samples = None
        self.clustered_points = None
        
        # Initialize plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Enter eps and min_spamples for DBSCAN Clustering')
        self.scatter = self.ax.scatter(self.x, self.y, self.z, c='grey')

        # Text boxes for parameters
        self.axbox_eps = plt.axes([0.1, 0.01, 0.35, 0.05])
        self.text_box_eps = TextBox(self.axbox_eps, 'Eps', initial="0.05")
        self.axbox_min_samples = plt.axes([0.55, 0.01, 0.35, 0.05])
        self.text_box_min_samples = TextBox(self.axbox_min_samples, 'Min Samples', initial="15")
        
        # Event handling
        self.text_box_eps.on_submit(self.update)
        self.text_box_min_samples.on_submit(self.update)

    def update(self, _):
        try:
            self.eps = float(self.text_box_eps.text)
            self.min_samples = int(self.text_box_min_samples.text)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return
        
        # Perform DBSCAN clustering
        X = np.column_stack((self.x, self.y, self.z))
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X)
        
        # Retrieve clustered points
        self.clustered_points = X[labels != -1]
        
        # Assign colors based on cluster labels
        colors = np.where(labels == -1, 'grey', 'blue')  # Grey for noise points, blue for clusters
        
        # Update scatter plot
        self.scatter.remove()
        self.scatter = self.ax.scatter(self.x, self.y, self.z, c=colors)
        self.fig.canvas.draw_idle()

    def get_clustered_points(self):
        return self.clustered_points

    def get_final_params(self):
        return self.eps, self.min_samples
class InteractiveDBSCAN2D:
    def __init__(self, df):
        self.x = df['distance'].values
        self.x = (self.x - self.x.mean())/self.x.std()
        self.y = df['MSL'].values
        self.y = (self.y - self.y.mean())/self.y.std()
        self.df = df
        self.eps = None
        self.min_samples = None
        self.df['clustered_points'] = 0
        
        # Initialize plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Enter eps and min_spamples for DBSCAN Clustering')
        self.scatter = self.ax.scatter(self.df['distance'], self.df['MSL'], c='grey')

        # Text boxes for parameters
        self.axbox_eps = plt.axes([0.1, 0.01, 0.35, 0.05])
        self.text_box_eps = TextBox(self.axbox_eps, 'Eps', initial="0.01")
        self.axbox_min_samples = plt.axes([0.55, 0.01, 0.35, 0.05])
        self.text_box_min_samples = TextBox(self.axbox_min_samples, 'Min Samples', initial="12")
        
        # Event handling
        self.text_box_eps.on_submit(self.update)
        self.text_box_min_samples.on_submit(self.update)

    def update(self, _):
        try:
            self.eps = float(self.text_box_eps.text)
            self.min_samples = int(self.text_box_min_samples.text)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return
        
        # Perform DBSCAN clustering
        X = np.column_stack((self.x, self.y))
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X)
        
        # Retrieve clustered points
        self.df['clustered_points'] = (labels!=-1)
        
        # Assign colors based on cluster labels
        colors = np.where(labels == -1, 'grey', 'blue')  # Grey for noise points, blue for clusters
        
        # Update scatter plot
        self.scatter.remove()
        self.scatter = self.ax.scatter(self.df['distance'], self.df['MSL'], c=colors)
        self.fig.canvas.draw_idle()

    def get_clustered_points_df(self):
        return self.df

    def get_final_params(self):
        return self.eps, self.min_samples
class InteractiveDataFramePlot:
    def __init__(self, df, x_column, y_column):
        self.df = df
        self.x_column = x_column
        self.y_column = y_column
        self.clicked_points = []

        # Scatter plot initialization
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter(self.df[self.x_column], self.df[self.y_column], marker='o', color='blue', label='Points')
        self.ax.set_xlabel(self.x_column)
        self.ax.set_ylabel(self.y_column)
        self.ax.set_title('Interactive Plot')

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()

    def plot_point(self, x, y):
        self.clicked_points.append((x, y))
        self.ax.scatter(x, y, marker='o', color='red')  # Highlight clicked point
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.button == 1:  # Left mouse button
            if event.xdata is not None and event.ydata is not None:
                self.plot_point(event.xdata, event.ydata)

    def get_clicked_points(self):
        plt.close()
        return self.clicked_points

# # Example usage
# if __name__ == '__main__':
#     # Example DataFrame
#     data = {
#         'X': [1, 2, 3, 4, 5],
#         'Y': [2, 3, 5, 7, 11]
#     }
#     df = pd.DataFrame(data)

#     interactive_plot = InteractiveDataFramePlot(df, 'X', 'Y')
#     clicked_points = interactive_plot.get_clicked_points()
#     print("Clicked points:", clicked_points)

# # Example usage with a DataFrame
# data = {
#     'lat_ph': np.random.rand(50),
#     'lon_ph': np.random.rand(50),
#     'h_ph': np.random.rand(50)
# }
# df = pd.DataFrame(data)

# # Create an instance of the InteractiveDBSCAN3D class with the DataFrame
# interactive_dbscan_3d = InteractiveDBSCAN3D(df)
# plt.show()

# # After closing the plot, retrieve final parameters and clustered points
# eps, min_samples = interactive_dbscan_3d.get_final_params()
# clustered_points = interactive_dbscan_3d.get_clustered_points()

# print(f"Final Eps: {eps}, Final Min Samples: {min_samples}")
# print(f"Clustered Points:\n{clustered_points}")








# # Example usage with a DataFrame
# data = {
#     'lat_ph': np.random.rand(500),
#     'lon_ph': np.random.rand(500),
#     'h_ph': np.random.rand(500)
# }
# df = pd.DataFrame(data)

# # Create an instance of the Interactive3DScatter class with the DataFrame
# interactive_plot = Interactive3DScatter(df)
# plt.show()

# # Retrieve final parameters after closing the plot window
# value1, value2 = interactive_plot.get_final_params()
# print(f"Final Value 1: {value1}, Final Value 2: {value2}")

# # # Create an instance of the InteractiveDBSCAN class with the DataFrame
# # interactive_dbscan = InteractiveDBSCAN(df)
# # plt.show()
# # # Retrieve final parameters after closing the plot window
# # eps, min_samples = interactive_dbscan.get_final_params()
# # print(f"Final Eps: {eps}, Final Min Samples: {min_samples}")
