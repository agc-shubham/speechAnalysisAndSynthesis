import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq
from matplotlib import pyplot as plt
import math
import scipy.signal as signal

samplerate=44100
time=0.010
samples=time*samplerate
amp=100

def dda_algo(x1,y1,x2,y2):
    dx=x2-x1
    dy=y2-y1
    if(abs(dx)>abs(dy)):
        steps=abs(dx)
    else:
        steps=abs(dy)
    xinc=dx/steps
    yinc=dy/steps
    x=x1
    y=y1
    r=list();
    for i in range(int(steps)):
        r.append(y)
        x=x+xinc
        y=y+yinc
        i=i+0.5
    return r

t1=np.linspace(0,3.5,100)
t2=np.linspace(3.5,7,100)
t3=np.linspace(7,7.5,100)
t4=np.linspace(7.5,8,100)
t5=np.linspace(8,8.5,100)
t6=np.linspace(8.5,9,100)
t7=np.linspace(9,9.5,100)
t8=np.linspace(9.5,10,100)
plt.plot(t1,dda_algo(0,0,3.5,100),'b-')
plt.plot(t2,dda_algo(3.5,100,7,0),'b-')
plt.plot(t3,dda_algo(7,0,7.5,-100),'b-')
plt.plot(t4,dda_algo(7.5,-100,8,0),'b-')
plt.plot(t5,dda_algo(8,0,8.5,100),'b-')
plt.plot(t6,dda_algo(8.5,100,9,0),'b-')
plt.plot(t7,dda_algo(9,0,9.5,-100),'b-')
plt.plot(t8,dda_algo(9.5,-100,10,0),'b-')
print(dda_algo(0,0,3.5,100))
plt.grid()
plt.show()

    
