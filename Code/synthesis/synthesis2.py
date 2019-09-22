import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from scipy.io import wavfile


Amplitude=100

x = np.arange(0, 34)
y = (Amplitude/33)*x
x1=np.arange(34,67)
y1=-(Amplitude/33)*(x-33)
x2=np.arange(67,100)
y2=-(Amplitude/33)*(x2-66)
x3=np.arange(100,133)
y3=(Amplitude/33)*(x3-100)-Amplitude
x4=np.arange(133,165)
y4=(Amplitude/33)*(x4-132)
x5=np.arange(165,199)
y5=-(Amplitude/33)*(x5-165)+Amplitude
x6=np.arange(199,231)
y6=-(Amplitude/33)*(x6-198)
x7=np.arange(231,265)
y7=(Amplitude/33)*(x7-231)-Amplitude
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

f = interpolate.interp1d([0,33,66,99,132,165,198,231,264], spline, kind='quadratic')
xnew = np.arange(0,265)
ynew = f(xnew)


plt.plot(np.arange(0,266),sharp_edge,'-',xnew,ynew,'--')
plt.plot(sharp_edge)
x_soft=np.arange(0,265)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)
ynew=np.append(ynew,ynew)



wavfile.write("C:\\Users\\AILAB-4\\Desktop\\synthesis_new.wav",44100,ynew)
plt.grid()
plt.show()
plt.plot(soft_edge)
plt.grid()
plt.show()
