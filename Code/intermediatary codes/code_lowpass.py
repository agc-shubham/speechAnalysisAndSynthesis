import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq
from matplotlib import pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 5
fs = 44100      # sample rate, Hz
cutoff = 900  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.figure(figsize=(30,10))
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, cutoff+200)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

samplerate, data = wavfile.read('C:\\Users\\AILAB-4\\Desktop\\wav files\\aa.wav')
data = data[:,0]
samples = data.shape[0]
times = np.arange(len(data))/float(samplerate)
T = samples/samplerate             # seconds
n = samples     # total number of samples
t = np.linspace(0, T, n, endpoint=False)
order = 5
fs = samplerate     # sample rate, Hz
   
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.figure(figsize=(30,10))
plt.plot(t, data, 'b-', label='data')
plt.xlim(0, T)
plt.title("ACTUAL INPUT DATA")
plt.xlabel('time (s)')
plt.grid()


plt.figure(figsize=(30,10))
plt.plot(t, y, 'b-', label='data')
plt.xlim(0, T)
plt.title("FILTERED DATA")
plt.xlabel('time (s)')
plt.grid()

datafft = fft(data)
filteredfft=fft(y)
fftabs = abs(datafft)
filterfftabs=abs(filteredfft)
freqs = fftfreq(samples,1/samplerate)

plt.figure(figsize=(30,10))
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)],'r')

plt.figure(figsize=(30,10))
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],filterfftabs[:int(freqs.size/2)],'g')

plt.show()
