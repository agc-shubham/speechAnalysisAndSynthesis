import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq
from matplotlib import pyplot as plt
import plotly.plotly as py
import peakutils


x=np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])
y=np.array([1,2,3,4,5])

length1=len(x)
length2=len(y)
l=length2

r=list();
period =length1/length2
print(period)
w=0
while True:
    n=0
    for i,j in zip(range(w,l) , range(length2)):
        n=n+x[i]*y[j]
    r.append(int(n))
    w=w+1
    l=l+1
    if(l>length1):
        break
    

plt.figure(figsize=(30,10))
plt.plot(r,'b-',)
plt.title("Cross Correlation 2")
plt.grid()

z=np.correlate(x,y,mode='valid')
plt.figure(figsize=(30,10))
plt.plot(z,'g-',)
plt.title("Cross Correlation ")
plt.grid()

r2=list();

maxm=max(r)
w=0
l=length2
while True:
    n=0
    for i,j in zip(range(w,l) , range(length2)):
        n=n+x[i]*y[j]
    r2.append((n/maxm))
    w=w+1
    l=l+1
    if(l>length1):
        break
    
for i in range(len(r2)):
    print(r2[i])

plt.figure(figsize=(30,10))
plt.plot(r2,'b-',)
plt.title("Maximised Cross Correlation 2")
plt.grid()


indices = peakutils.indexes(r2, thres=0.70, min_dist=0.1)
trace = go.Scatter(
    x=[j for j in range(len(r2))],
    y=r2,
    mode='lines',
    name='Original Plot'
)

trace2 = go.Scatter(
    x=indices,
    y=[r2[j] for j in indices],
    mode='markers',
    marker=dict(
        size=8,
        color='rgb(255,0,0)',
        symbol='cross'
    ),
    name='Detected Peaks'
)

data = [trace, trace2]
py.iplot(data, filename='milk-production-plot-with-higher-peaks')


plt.show()
