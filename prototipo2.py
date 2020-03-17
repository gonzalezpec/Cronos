import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
from math import pi
from statistics import stdev 

with open("C:/Users/hp/Desktop/spec/spec183188.txt", 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    y = [float(line.split()[1]) for line in lines]

x = np.asarray(x)   
y = np.asarray(y)

y = y[(x>4835) & (x<5020)]
x = x[(x>4835) & (x<5020)]

'''#graficamos para comprobar que el arreglo fue un exito.
plt.plot(x,y)
plt.show()'''

#valores iniciales 
#Hbeta
cH=500.0
sH=0.75
uH=4861.0
#Oiii1
cO1=500.0
sO1=0.75
uO1=4959.0
#Oii2
cO2=500.0
sO2=0.75
uO2=5007.0

#buscamos centro variable de uH
yH = y[(x>(uH-10)) & (x<(uH+10))]
xH = x[(x>(uH-10)) & (x<(uH+10))]
uH=xH[yH==max(yH)]

#buscamos centro variable de uO2
yO2 = y[(x>(uO2-10)) & (x<(uO2+10))]
xO2 = x[(x>(uO2-10)) & (x<(uO2+10))]
uO2=xO2[yO2==max(yO2)]

#buscamos centro variable de u2. Siguiendo la misma filosofia de Oiii1 y
#Oii2, acoplamos el oiii1 al oiii2
uO1=(4959.0/5007.0)*uO2

plt.plot(x,y)
plt.axvline(x=uH,color="black")
plt.axvline(x=uO1,color="green")
plt.axvline(x=uO2,color="red")
plt.show()

class gauss(object):
	def __init__(self,r,cH,sH,uH,cO1,sO1,uO1,cO2,sO2,uO2):
		#FUNCION GAUSSIANA 
		gH = (cH/(sH*math.sqrt(2.0*pi)))*np.exp((-(r-uH)**2)/(2.0*sH**2))
		gO1 = (cO1/(sO1*math.sqrt(2.0*pi)))*np.exp((-(r-uO1)**2)/(2.0*sO1**2))
		gO2 = (cO2/(sO2*math.sqrt(2.0*pi)))*np.exp((-(r-uO2)**2)/(2.0*sO2**2))

		#DERIVADA PARCIAL CONRESPECTO A c
		gcH = (1/(sH*math.sqrt(2.0*pi)))*np.exp((-(r-uH)**2)/(2.0*sH**2))
		gcO1 = (1/(sO1*math.sqrt(2.0*pi)))*np.exp((-(r-uO1)**2)/(2.0*sO1*2))
		gcO2 = (1/(sO2*math.sqrt(2.0*pi)))*np.exp((-(r-uO2)**2)/(2.0*sO2**2))

		#DERIVADA PARCIAL CON RESPECTO A s
		gsH = (cH*(-(np.exp((-(r-uH)**2)/(2.0*sH**2)))*sH**2+(np.exp((-(r-uH)**2)/(2.0*sH**2)))*(r-uH)**2))/(math.sqrt(2.0*pi)*sH**4)
		gsO1 = (cO1*(-(np.exp((-(r-uO1)**2)/(2.0*sO1**2)))*sO2**2+(np.exp((-(r-uO1)**2)/(2.0*sO1**2)))*(r-uO1)**2))/(math.sqrt(2.0*pi)*sO1**4)
		gsO2 = (cO2*(-(np.exp((-(r-uO2)**2)/(2.0*sO2**2)))*sO2**2+(np.exp((-(r-uO2)**2)/(2.0*sO2**2)))*(r-uO2)**2))/(math.sqrt(2.0*pi)*sO2**4)
		
		#DERIVADA PARCIAL CONRESPECTO A u
		guH = cH*(r-uH)/(sH**3*math.sqrt(2.0*pi))*np.exp((-(r-uH)**2)/(2.0*sH**2))
		guO1 = cO1*(r-uO1)/(sO1**3*math.sqrt(2.0*pi))*np.exp((-(r-uO1)**2)/(2.0*sO1**2))
		guO2 = cO2*(r-uO2)/(sO2**3*math.sqrt(2.0*pi))*np.exp((-(r-uO2)**2)/(2.0*sO2**2))

		self.g1 = gH
		self.g2 = gO1
		self.g3 = gO2
		self.g1c = gcH
		self.g2c = gcO1
		self.g3c = gcO2
		self.g1s = gsH
		self.g2s = gsO1
		self.g3s = gsO2
		self.g1u = guH
		self.g2u = guO1
		self.g3u = guO2

res=gauss(x,cH,sH,uH,cO1,sO1,uO1,cO2,sO2,uO2)
plt.plot(x, res.g1, color='black')
plt.plot(x, res.g2, color='black')
plt.plot(x, res.g3, color='black')
plt.plot(x, res.g1c, color='red')
plt.plot(x, res.g2c, color='red')
plt.plot(x, res.g3c, color='red')
plt.plot(x, res.g1s, color='blue')
plt.plot(x, res.g2s, color='blue')
plt.plot(x, res.g3s, color='blue')
plt.plot(x, res.g1u, color='green')
plt.plot(x, res.g2u, color='green')
plt.plot(x, res.g3u, color='green')
plt.show()

f = gauss(x,cH,sH,uH,cO1,sO1,uO1,cO2,sO2,uO2)
dy = y-(f.g1+f.g2+f.g3)
s0 = stdev(dy)
sn = 10*s0
k = 0.2
l = 0
plt.plot(x,dy)
plt.show()
while abs(sn-s0)>(0.001*s0):
	l = l+1
	s0 = sn
	jact = np.array([f.g1c, f.g1s, f.g1u, f.g2c, f.g2s, f.g2u, f.g3c, f.g3s, f.g3u])	
	jac = jact.transpose()
	A = np.matmul(jact,jac)
	Ainv = np.linalg.inv(A)
	B = np.matmul(jact,dy)
	db = np.matmul(Ainv,B)
	cH = cH + k * db[0]
	sH = sH + k * db[1]
	uH = uH + k * db[2]
	cO1 = cO1 + k * db[3]
	sO1 = sO1 + k * db[4]
	uO1 = uO1 + k * db[5]
	cO2 = cO2 + k * db[6]
	sO2 = sO2 + k * db[7]
	uO2 = uO2 + k * db[8]
	f = gauss(x,cH,sH,uH,cO1,sO1,uO1,cO2,sO2,uO2)
	dy = y-(f.g1+f.g2+f.g3)
	sn = stdev(dy)
plt.plot(x,dy)

'''xx = np.arange(x[0],x[len(x)-1],0.1)
yy = gauss(x,cH,sH,uH,cO1,sO1,uO1,cO2,sO2,uO2)
plt.plot(xx, yy)'''

plt.axvline(x = uH, color = 'red')
plt.axvline(x = uO1, color = 'red')
plt.axvline(x = uO2, color = 'red')
plt.show()