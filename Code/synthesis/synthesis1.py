import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import wavfile



Amplitude=100

x = np.arange(0, 45)
y = (Amplitude/44)*x
x1=np.arange(44,89)
y1=-(Amplitude/44)*(x-44)
x2=np.arange(88,111)
y2=-(Amplitude/22)*(x2-88)
x3=np.arange(110,133)
y3=(Amplitude/22)*(x3-110)-Amplitude
x4=np.arange(132,155)
y4=(Amplitude/22)*(x4-132)
x5=np.arange(154,177)
y5=-(Amplitude/22)*(x5-154)+Amplitude
x6=np.arange(176,221)
y6=-(Amplitude/44)*(x6-176)
x7=np.arange(220,265)
y7=(Amplitude/44)*(x7-220)-Amplitude
spline = list();
sharp_edge=np.array([])
sharp_edge=np.append(sharp_edge,y)
spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])

sharp_edge=np.append(sharp_edge,y1)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y2)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y3)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y4)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y5)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y6)
#spline.append(sharp_edge[0])
spline.append(sharp_edge[-1])
sharp_edge=np.append(sharp_edge,y7)
spline.append(sharp_edge[-1])
for i in range(len(spline)):
    print(spline[i])


soft_edge = butter_lowpass_filter(sharp_edge, 1000, 44100, 5)
wavfile.write("C:\\Users\\AILAB-4\\Desktop\\synthesis1.wav",44100,sharp_edge)


f = interpolate.interp1d([0,44,88,110,132,154,176,220,264], spline, kind='quadratic')
xnew = np.arange(0,265)
ynew = f(xnew)


plt.plot(np.arange(0,272),sharp_edge,'-',xnew,ynew,'--')
plt.plot(sharp_edge)
x_soft=np.arange(0,265)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)



wavfile.write("C:\\Users\\AILAB-4\\Desktop\\synthesis12.wav",44100,ynew)
plt.grid()
plt.show()
plt.plot(soft_edge)
plt.grid()
plt.show()
