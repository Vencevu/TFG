import csv
from decimal import Decimal
import matplotlib.pyplot as plt

dist_csv = open('csv/validate/Acc1.csv', 'r')
data = csv.reader(dist_csv,delimiter=';')

x=[]
y=[]

for row in data:
    d = row
    num = float(d[1])

    y.append(num)
    x.append(int(d[0]))
    


plt.plot(x, y)
plt.xlabel("Episodios")
plt.ylabel("Precisión")
plt.savefig('Acc.png')

dist_csv.close()