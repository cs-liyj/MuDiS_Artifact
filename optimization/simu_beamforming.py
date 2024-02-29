import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import librosa
import soundfile as sf
import scipy.signal
import scipy.io.wavfile as wavfile
import wave
import argparse
d=0.008
c=344.0
position=[-7*d/2,-5*d/2,-3*d/2,-d/2,d/2,3*d/2,5*d/2,7*d/2]


def Read_8channel_audio(output_path):
    sig,sr=librosa.load(output_path,sr=None, mono=False)

    return sig,sr

def Simu_Receive(input_audio,sr,theta):

    output_audio = np.zeros((8, len(input_audio[0])))
    frequencies=[18000,24000]
    idx1=int(frequencies[0]*len(input_audio[0])/sr)
    idx2=int(frequencies[1]*len(input_audio[0])/sr)

    for i in range(8):
        temp=input_audio[i]
        input_spec=np.fft.rfft(temp)
        f=np.arange(int(len(temp)//2))*(sr/len(temp))
        output_spec=input_spec

        for j in range(idx1,idx2):
            select_f=f[j]
            wavelength=c/select_f
            phase_diff= 2 * np.pi* i*d /wavelength * np.sin(np.radians(theta))

            output_spec[j]= input_spec[j] * np.exp(1j * phase_diff)

        output_audio[i]=np.fft.irfft(output_spec)

    output=np.sum(output_audio,axis=0)

    return output, sr

def Draw_beampattern(path):
    theta_list=np.arange(-90,91,1)
    # theta_list=np.arange(0,185,10)
    # print(theta_list)
    max_amp_list=[]

    input_audio,sr=Read_8channel_audio(path)
    for theta in theta_list:
        rec_audio,sr=Simu_Receive(input_audio,sr,theta)
        f,p1,phi, max_f, max_amp,max_phi=FFT(rec_audio,sr)
        print(theta,max_f,max_amp)

        max_amp_list.append(max_amp)

    plt.plot(theta_list,max_amp_list)


def Get_audio_Lowfreq(path,theta,outputfolder,lowfreq=True):
    input_audio,sr=Read_8channel_audio(path)
    output,sr=Simu_Receive(input_audio,sr,theta)
    output = output / np.max(output)

    if lowfreq:
        rec=output*output

        cut_Freq=8000
        cutoff_freq = cut_Freq / sr
        b, a = scipy.signal.butter(8, cutoff_freq, 'lowpass')

        filtered_data = scipy.signal.filtfilt(b, a, rec)
        filtered_data  = filtered_data - np.average(filtered_data)
        filtered_data = filtered_data/np.max(filtered_data)

        filename=path.split('/')[-1]
        outputpath=outputfolder+'LowFreq_'+str(theta)+'_'+filename

        sf.write(outputpath, filtered_data, sr)

    else:
        output=output/8
        filename=path.split('/')[-1]
        outputpath=outputfolder+'HighFreq_'+str(theta)+'_'+filename

        sf.write(outputpath,output,sr)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generates BF pattern.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--angles", type=str, default="0,40")
    parser.add_argument("--channel8_wavpath", type=str, default="./modulated_audios_disopt/woman1dsb_0_man1dsb_40_8channel_optnullsteeringphaseshifted_d_0.008_distortionopted.wav")
    args = parser.parse_args()

    phis_str = args.angles.split(",")
    target_angles = [int(phi) for phi in phis_str]

    for target_angle in target_angles:
        Get_audio_Lowfreq(args.channel8_wavpath, target_angle, "simu_lowfreq_audios/",lowfreq=True)
