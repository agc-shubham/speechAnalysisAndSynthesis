import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq
from matplotlib import pyplot as plt
import math
import scipy.signal as signal

def butter_lowpass(cutoff, fs, order=5): 
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):        #filtering
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)                                  #filtered array
    return y

def window_creation(y,t,delay,samplerate):
    sample=t*samplerate
    r=list();
    for i in range(int(sample)):
        n=y[i+delay]
        r.append(n)
    return r

def wave_correlation(y,x):
    len_y=len(y)
    len_x=len(x)
    r=list();
    if(len_y==len_x):                                                                                             
        for i in range(len_y):
            n=0
            for j in range(i,len_y):
                n=n+(y[j]*y[j-i])
            r.append(int(n))
    else:                                          
        l=len_x
        w=0
        while True:
            n=0
            for i,j in zip(range(w,l) , range(len_x)):
                n=n+y[i]*x[j]
            r.append(int(n))
            w=w+1
            l=l+1
            if(l>len_y):
                break

    return r

def wave_normalization(y,x):
    len_y=len(y)
    len_x=len(x)
    a=0
    for i in range(len_y):
        a=a+math.pow(y[i],2)
    b=0
    for j in range(len_x):
        b=b+math.pow(x[j],2)
    N=np.sqrt(a*b)
    r=wave_correlation(y,x)
    r1=list();
    for i in range(len(r)):
        n=r[i]/N
        r1.append(n)
    return r1

def wave_maximization(y,x):
    r1=list();
    r=wave_correlation(y,x)
    maxm=max(r)
    for i in range(len(r)):
        n=r[i]/maxm
        r1.append(n)
    return r1

def peak_finding(r):
    peakidx = signal.find_peaks_cwt(r,np.arange(1,2))         #x-axis array of the peaks
    peak=list();                                                 #y-axis values of the peaks
    n=0
    for i in range(len(peakidx)):
        n=r[peakidx[i]]
        peak.append(n)
    return peak

def peak_finding1(r):
    peakidx = signal.find_peaks_cwt(r,np.arange(1,2))         #x-axis array of the peaks
    peak=list();                                                 #y-axis values of the peaks
    n=0
    for i in range(len(peakidx)):
        n=r[peakidx[i]]
        peak.append(n)
    return peak,peakidx

def remove_below70(x):
    y_axis,x_axis=peak_finding1(x)
    l=list();
    p=list();
    for i in range(len(y_axis)):
        if(y_axis[i]>0.72):
            m=y_axis[i]
            n=x_axis[i]
            l.append(m)
            p.append(n)
    return l,p

def zero_crossing(x):
    zc=0
    for i in range(1,len(x)):
        if(x[i-1]>0 and x[i]<0):
            zc=zc+1
        elif(x[i-1]<0 and x[i]>0):
            zc=zc+1
        else:
            zc =zc
    return zc        

def graph_plot(x,y,s_title,color):
    plt.figure(figsize=(30,10))
    plt.plot(x,y,color)
    plt.title(s_title)
    plt.grid()

def graph_plot1(y,s_title,color):
    plt.figure(figsize=(30,10))
    plt.plot(y,color)
    plt.title(s_title)
    plt.grid()

name=input("enter the name of the file")
name=name+".wav"
samplerate, data = wavfile.read('C:\\Users\\AILAB-4\\Desktop\\wav files\\'+name)
data = data[:,0]                                       #real data
samples = data.shape[0]
times = np.arange(len(data))/float(samplerate)
T = samples/samplerate                                 # seconds
n = samples                                            # total number of samples
t = np.linspace(0, T, n, endpoint=False)
order = 5
fs = samplerate                                        # sample rate, Hz
cutoff = 900


b, a = butter_lowpass(cutoff, fs, order)
y = butter_lowpass_filter(data, cutoff, fs, order)
datafft = fft(data)
filteredfft=fft(y)
fftabs = abs(datafft)
filterfftabs=abs(filteredfft)
freqs = fftfreq(samples,1/samplerate)

#template formation for sliced correlation and time periiod calculation

t_template=0.015
t_window=0.050
sample_template= t_template*samplerate
length=int(samples/sample_template)
delay=0
timePeriod=list();
zerocount=list();
for i in range(length):
    y_new=window_creation(y,t_window,delay,samplerate)
    x=window_creation(y,t_template,delay,samplerate)
    y_indice, x_indice=remove_below70(wave_maximization(y_new,x))
    n=(x_indice[1]-x_indice[0])/samplerate
    timePeriod.append(n)
    delay=x_indice[-1]
    
print(timePeriod)

#zero counts for calculated time period of above samples 
cntd=0
for i in range(len(timePeriod)):
    a=list();
    b=samplerate*timePeriod[i]
    for j in range(int(b)):
        m=y[j+int(cntd)]
        a.append(m)
    n=zero_crossing(a)
    zerocount.append(n)
    cntd=cntd+b
print(zerocount)

#Visualizations

plt.hist(zerocount)
plt.ylabel('No of times')


graph_plot(t, data , "ACTUAL DATA",'b-')

graph_plot(t, y , "FILTERED DATA",'g-')

graph_plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)],"ACTUAL FREQUENCY PLOTS",'r-')
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.xlabel("Frequency(Hz)")

graph_plot(freqs[:int(freqs.size/2)],filterfftabs[:int(freqs.size/2)],"FILTERED FREQUENCY PLOTS",'b-')
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.xlabel("Frequency(Hz)")


graph_plot(t1, wave_correlation(y_new,y_new) , "SAMPLED AUTO CORRELATION",'g-')

graph_plot(t, wave_correlation(y,y) , "AUTO CORRELATION",'g-')

graph_plot1(wave_correlation(y,x) , "CROSS CORRELATION",'r-')

graph_plot1(wave_correlation(y_new,x) , "SAMPLED CROSS CORRELATION",'g-')

graph_plot1(wave_normalization(y,y) , "AUTO_CORRELATION_NORMALISED",'b-')

graph_plot1(wave_normalization(y,x) , "CROSSED_CORRELATION_NORMALISED",'g-')

graph_plot1(wave_maximization(y_new,x) , "CROSSED_CORRELATION_MAXIMISED",'g-')

y_indice, x_indice=remove_below70(wave_maximization(y_new,x))
print("time period ={}".format((x_indice[1]-x_indice[0])/samplerate))
print(x_indice[-1])
graph_plot(x_indice, y_indice , "ABOVE 70",'g-')

graph_plot1(peak_finding(wave_correlation(y_new,x)) , "PEAK OVER CROSS CORRELATION",'g-')

graph_plot1(peak_finding(wave_normalization(y_new,x)) , "PEAK OVER NORMALIZATION",'g-')


plt.show()






    

