import csv
import matplotlib.pyplot as plt

dist_csv = open('csv/validate/Distances.csv', 'r')
data = csv.reader(dist_csv)

x_axis_time = []
x_axis_goal = []
x_axis_col = []
y_axis_time = []
y_axis_col = []
y_axis_goal = []

for row in data:
    d = row[0].split(';')
    if d[2] == 'Tiempo':
        y_axis_time.append(d[1])
        x_axis_time.append(d[0])
    elif d[2] == 'Colision':
        y_axis_col.append(d[1])
        x_axis_col.append(d[0])
    elif d[2] == 'Objetivo':
        y_axis_goal.append(d[1])
        x_axis_goal.append(d[0])

print(len(x_axis_time))
print(len(x_axis_col))
print(len(x_axis_goal))

plt.scatter(x_axis_time, y_axis_time, label="time reset")
plt.scatter(x_axis_col, y_axis_col, label="collision reset")
plt.scatter(x_axis_goal, y_axis_goal, label="goal")
plt.legend(loc="upper left")
plt.xlabel("Episodios")
plt.ylabel("Distancia al objetivo")
plt.savefig('Distancias.png')

dist_csv.close()