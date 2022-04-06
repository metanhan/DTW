# Kütüphaneleri dahil et.

import pyaudio
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
import os, glob
from scipy import stats


# Komut kayıt et.



for i in range(5):

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = f"C:/Users/Mehmethan/Desktop/dtw 2/komutlar/komut{i}.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    


# Tek kullanımlık 2 saniyelik istek ses kaydı al.

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "istek.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()



# Klasördeki bütün ses dosyalarını bir listeye atadık, index numaraları ile çağırılabilir haldeler.

zero = []
path = "C:/Users/Mehmethan/Desktop/dtw 2/komutlar/"
files = os.listdir(path)

for filename in glob.glob(os.path.join(path, '*.wav')):
    samplerate, data = read(filename)
    zero.append(data)

readIstek=read('istek.wav')
istek=readIstek[1]

# Seslerin grafiklerini çizdir.

plt.plot(zero[0],"r-")
plt.title("Komut 1")
plt.show()

plt.plot(zero[1],"r-")
plt.title("Komut 2")
plt.show()

plt.plot(zero[2],"r-")
plt.title("Komut 3")
plt.show()

plt.plot(zero[3],"r-")
plt.title("Komut 4")
plt.show()

plt.plot(zero[4],"r-")
plt.title("Komut 5")
plt.show()

plt.plot(istek,'r-')
plt.title("İstek")
plt.show()



# Seste başta ve sonda bulunan boşluk kısımlarını at.

kirp=[]
son=[]

for i in range(len(zero)):
    kirp.append(np.where((-10<= zero[i]) & (zero[i]<=10)))
    son.append(np.delete(zero[i],kirp[i]))
   
plt.plot(son[0],"r-")
plt.title("Komut 1")
plt.show()

plt.plot(son[1],"r-")
plt.title("Komut 2")
plt.show()

plt.plot(son[2],"r-")
plt.title("Komut 3")
plt.show()

plt.plot(son[3],"r-")
plt.title("Komut 4")
plt.show()

plt.plot(son[4],"r-")
plt.title("Komut 5")
plt.show()

kirp2=np.where((-10<= istek) & (istek<=10))

for i in range(len(kirp2)):
    ekle=np.delete(istek,kirp2[i])
    plt.plot(ekle,'r', label='x')
    plt.title("İstek")
    plt.show()
    


# Verileri daha kolay işlemek için daha küçük parçalara böl.

islem=[]

for i in range(len(son)):
    islem.append(np.array_split(son[i],10))

istek_split=[]
istek_split.append(np.array_split(ekle,10))
   


# Main ve İstek değişkenlerini belirle indislerden.

for j in range(len(islem)):
    main_islem=islem[j]
    print(f"işlem {j}")
    for k in range(len(main_islem)):
        main=main_islem[k]
        istek_islem=istek_split[0]
        istek=istek_islem[k]
        
        print(f"main {k}")
        print(f"istek {k}")

#########################################
        
        distances = np.zeros((len(istek), len(main)))
      
        
        for i in range(len(istek)):
            for j in range(len(main)):
                distances[i,j] = (main[j]-istek[i])**2  
      
                
        def distance_cost_plot(distances):
            im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
            plt.gca().invert_yaxis()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid()
     
            
        distance_cost_plot(distances)
        
        accumulated_cost = np.zeros((len(istek), len(main)))
        accumulated_cost[0,0] = distances[0,0]
        
        for i in range(1, len(main)):
            accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]    
        
        for i in range(1, len(istek)):
            accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]    
            
        for i in range(1, len(istek)):
            for j in range(1, len(main)):
                accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], 
                                             accumulated_cost[i-1, j], 
                                             accumulated_cost[i, j-1]) + distances[i, j]
        
        distance_cost_plot(accumulated_cost)

        
        path = [[len(main)-1, len(istek)-1]]
        i = len(istek)-1
        j = len(main)-1
        while i>0 and j>0:
            if i==0:
                j = j - 1
            elif j==0:
                i = i - 1
            else:
                if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], 
                                                   accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                    i = i - 1
                elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], 
                                                     accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                    j = j-1
                else:
                    i = i - 1
                    j= j- 1
            path.append([j, i])
        path.append([0,0])
        
        
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        
        distance_cost_plot(accumulated_cost)
        plt.plot(path_x, path_y);
        plt.show()
        
        def path_cost(x, y, accumulated_cost, distances):
            path = [[len(x)-1, len(y)-1]]
            cost = 0
            i = len(y)-1
            j = len(x)-1
            while i>0 and j>0:
                if i==0:
                    j = j - 1
                elif j==0:
                    i = i - 1
                else:
                    if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], 
                                                       accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        i = i - 1
                    elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], 
                                                         accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        j = j-1
                    else:
                        i = i - 1
                        j= j- 1
                path.append([j, i])
            path.append([0,0])
            for [y, x] in path:
                cost = cost +distances[x, y]
            return path, cost
    

# Çalıştır.

        path, cost = path_cost(main, istek, accumulated_cost, distances)
        print(cost,'\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


