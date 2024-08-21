import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display
from tkinter import TclError
import tensorflow as tf
from tensorflow import keras
from scipy.io import wavfile
from playsound import playsound
import sounddevice as sd
import os

import time

plt.switch_backend('tkagg')

# CHANGE DURATION AND MFCC LENGTH TO MATCH INPUT DATA SIZE
duration = 85980
mfcc_length = 672


div_chunk = 2
CHUNK = duration // div_chunk   
slow = 0.67

n_fft = 512
hop_length = 128
n_mfcc = 12
sr = 22050

model = keras.models.load_model('models/best_model')


# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True, output = True, frames_per_buffer=CHUNK)

# basic formatting for the axes
ax.set_title('LIVE MFCC')

        
# show the plot
plt.show(block=False)

print('stream started')

while True:

    # initialize the buffer
    chunk_list = np.empty((div_chunk, CHUNK), np.float32)
    #print(chunk_list.shape)
    wait_fill_buffer = 0
    stressed = False
    index = 0
    
    while index < div_chunk:

        data = stream.read(CHUNK)  
        chunk_list[index] = np.frombuffer(data, dtype=np.float32)
        index = index + 1
        
    index = div_chunk - 1
    stress_meter = 0

    
    
    while True: 

        #sample new chunk
        if stressed and wait_fill_buffer == div_chunk:
            data = stream.read(CHUNK)
            data = np.frombuffer(data, dtype=np.float32)
            chunk_list = np.concatenate((chunk_list, data.reshape(1,-1)), axis = 0)
            #print(chunk_list.shape)
            index += 1
        else:
            chunk_list = np.roll(chunk_list,-chunk_list.shape[1])
            data = stream.read(CHUNK)  
            data = np.frombuffer(data, dtype=np.float32)
            
            chunk_list[index] = data
            #print(index)
        
        

        #convert MFCC
        data_np = chunk_list.flatten()
        mfcc = librosa.feature.mfcc(y = data_np[-duration:], n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc)

        print()
        #print(data.std())
        #print(data_np[-duration:].std())
                               
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Current Time: ", current_time)
                
        # If STRESSED condition
        if stressed:
            print('***STRESSED***')      
            print('stress_meter:', stress_meter, 'seconds')              
            wait_fill_buffer += 1   
            if wait_fill_buffer == div_chunk + 1:
                wait_fill_buffer = 0
                slowdown = librosa.effects.time_stretch(data_np, rate=slow, n_fft = n_fft)
                sd.play(slowdown, 22050)



            if stress_meter <= 0:
                stress_meter = 0
                stressed = False
                sd.stop()
                break;
                
	
 
        # Silence 
        if (data_np[-duration:].std() < 2e-3):
            print('silence')
            if stressed:
                stress_meter -= 2
            else:
                stress_meter -= 5
            if stress_meter < 0:
                stress_meter = 0
                   
        else:
            
            prediction = model.predict(mfcc.reshape(1,n_mfcc,mfcc_length,1))
            
            # threshold for prediction confidence
            if prediction > 0.3 and prediction < 0.7:
                y_pred = -1
            else:
                y_pred = prediction.round()

            if y_pred == -1:
                print('unknown')
                if stressed:
                    stress_meter -= 2
                else:
                    stress_meter -= 5
                if stress_meter < 0:
                    stress_meter = 0
                
            elif y_pred == 0:
                print('neutral {}'.format(1 - prediction[0,0]))
                if stressed:
                    stress_meter -= 2
                else:
                    stress_meter -= 5
                if stress_meter < 0:
                    stress_meter = 0
                        
            elif y_pred == 1:
                if stressed:
                    stress_meter += 2
                else:
                    stress_meter+=10
                    
                print('stress {}'.format(prediction[0,0]))
                if not stressed and stress_meter >= 15:
                    stress_meter = 15
                    slowdown = librosa.effects.time_stretch(data_np, rate=slow, n_fft = n_fft)
                    sd.play(slowdown, 22050)
                    stressed = True
                    print ('ENTERING STRESSED MODE')

            
        librosa.display.specshow(mfcc, x_axis = "s", y_axis = "mel", sr = sr, hop_length = hop_length)
        # update figure canvas
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()	
           

        except TclError:
            break