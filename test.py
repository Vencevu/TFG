import csv
from decimal import Decimal
import matplotlib.pyplot as plt

dist_csv = open('csv/validate/Loss1.csv', 'r')
data = csv.reader(dist_csv,delimiter=';')

x=[]
y=[]

for row in data:
    d = row
    num = d[1]

    num = num.replace('.', '', num.count('.') - 1)
    num = float(num)
    y.append(num)
    x.append(int(d[0]))
    


plt.plot(x, y)
plt.yscale('log')
plt.xlabel("Episodios")
plt.ylabel("Pérdida")
plt.savefig('Loss.png')

dist_csv.close()