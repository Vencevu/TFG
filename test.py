import csv
from decimal import Decimal
import matplotlib.pyplot as plt

dist_csv = open('csv/validate/Distances.csv', 'r')
data = csv.reader(dist_csv,delimiter=';')

x_axis_time = []
x_axis_goal = []
x_axis_col = []
y_axis_time = []
y_axis_col = []
y_axis_goal = []

for row in data:
    d = row
    
    num = float(d[1].replace(',', '.'))
    num = float(num)

    if d[2] == 'Tiempo':
        y_axis_time.append(num)
        x_axis_time.append(int(d[0]))
    elif d[2] == 'Colision':
        y_axis_col.append(num)
        x_axis_col.append(int(d[0]))
    elif d[2] == 'Objetivo':
        y_axis_goal.append(num)
        x_axis_goal.append(int(d[0]))
    


plt.scatter(x_axis_time, y_axis_time, label="time reset")
plt.scatter(x_axis_col, y_axis_col, label="collision reset")
plt.scatter(x_axis_goal, y_axis_goal, label="goal")
plt.legend(loc="upper left")
plt.xlabel("Episodios")
plt.ylabel("Distancia al objetivo")
plt.savefig('Distancias.png')

dist_csv.close()