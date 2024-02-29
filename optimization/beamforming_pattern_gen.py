#!/usr/bin/env python3

"""This module plots and outputs beamforming pattern.

Example usage: python3 beamforming_pattern_gen.py

Author: CGrassin (http://charleslabs.fr)
License: MIT
"""
import wave
from cProfile import label
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from numpy.fft import fft, ifft, rfft, irfft
import scipy.fftpack as fftp
import librosa
import soundfile
from tqdm import tqdm
import copy
import os
import librosa
import json
from scipy import signal
import scipy.io.wavfile as wavfile
import wave as wv
import subprocess
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter, resample_poly, hilbert

def load_json_file(json_file_name):
    f = open(json_file_name, 'r')
    content = f.read()
    f.close()
    return json.loads(content)

def dBtoLinear(db):
    """Converts a log value to a linear value."""
    return 10**(db/20);

def lineartodB(lin):
    """Converts a log value to a linear value."""
    return 20*math.log10(lin);

# Get amplitude law
def get_amplitude_law(N, law = 'constant', minAmp = 1):
    """Computes an amplitude law given N (number of elements),
    law (type of law) and minAmp (minimum amplitude).
    """
    amp_law = []
    
    chebshev_win = [0.1461,0.4179,0.7594,1.0000,1.0000,0.7594,0.4179,0.1461]

    for n in range(N):
        if law == 'constant':
            amp_law.append(1);
        elif law == 'linear':
            beta = 0 if N%2!=0 else (1-minAmp)/(N-1)
            amp_law.append((minAmp-1-beta) * 2/(N-1) * abs(n - (N-1) / 2) + 1 + beta);
        elif law == 'log_linear':
            beta = 0 if N%2!=0 else (1-lineartodB(minAmp))/(N-1)
            amp_law.append(dBtoLinear((lineartodB(minAmp)-beta) * 2/(N-1) * abs(n-(N-1)/2) + beta));
        elif law == 'poly2':
            beta = 0 if N%2!=0 else (1-minAmp)/(N-1)
            amp_law.append((minAmp-1-beta) * (2/(N-1))**2 * (n-(N-1)/2)**2 + 1 + beta**2);
        elif law == 'poly3':
            beta = 0 if N%2!=0 else (1-minAmp)/(N-1)
            amp_law.append((minAmp-1-beta) * (2/(N-1))**3 * abs(n-(N-1)/2)**3 + 1 + beta**3);
        elif law == 'chebshev':
            amp_law.append(chebshev_win[n]) 
    return amp_law;

# Get phase law
def get_phase_law(N, d, wavelength, phi):
    """Computes a phase law given N (number of elements),
    d (spacing between elements in m), wavelength (in m)
    and phi (beam steering angle).
    """
    phase_law = [];
    for n in range(N):
        # 2pi * n * d / lambda
        phase_law.append(-2 * math.pi * n * d / wavelength * math.sin(math.radians(phi)));
    return phase_law;   

# Get wideband phase law
def get_wideband_phase_law(N, d, wavelength, phi, f_c, bandwidth, bins_num):
    """Computes a phase law given N (number of elements),
    d (spacing between elements in m), wavelength (in m)
    and phi (beam steering angle).
    """
    delta_f = bandwidth / bins_num
    f_m_list = []
    for m in range(int(bins_num / 2), bins_num):
        f_m_list.append(f_c + (m - bins_num) * delta_f)
    for m in range(int(bins_num / 2)):
        f_m_list.append(f_c + m * delta_f)
    phase_law = []
    for f_m in f_m_list:
        now_wavelength = 344 / f_m
        f_m_phase_law = []
        for n in range(N):
            # 2pi * n * d / lambda 
            f_m_phase_law.append(-2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi)))
        phase_law.append(f_m_phase_law)
    return f_m_list, phase_law

# Compute antenna pattern
def get_pattern(N, d, wavelength, phi, amplitude_law, minimum_amplitude, logScale=True):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    phase_law = get_phase_law(N, d, wavelength, phi);
    
    theta = np.arange(-90,90,0.1);
    mag = []
    for i in theta:
        im=0
        re=0
        # Phase shift due to off-boresight angle
        psi = 2 * math.pi * d / wavelength * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += amp_law[n] * math.sin(n*psi + phase_law[n])
            re += amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude = math.sqrt(re**2 + im**2)/N
        if logScale:
            magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

# Compute antenna pattern
def get_pattern_phase(N, d, wavelength, phi, amplitude_law, minimum_amplitude, logScale=True):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    phase_law = get_phase_law(N, d, wavelength, phi);
    
    theta = np.arange(-90,90,0.1);
    phases = []
    for i in theta:
        im=0
        re=0
        # Phase shift due to off-boresight angle
        psi = 2 * math.pi * d / wavelength * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += amp_law[n] * math.sin(n*psi + phase_law[n])
            re += amp_law[n] * math.cos(n*psi + phase_law[n])
        phase = np.angle(re + im * 1j)
        phases.append(phase)
        
    return theta, phases, amp_law, phase_law

def get_tmp_pattern(N, d, wavelength, phi, amplitude_law, minimum_amplitude, logScale=True):
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    phase_law = get_phase_law(N, d, wavelength, phi);
    
    theta = np.arange(-90,90,0.1)
    mag = []
    for i in theta:
        im=0
        re=0
        # Phase shift due to off-boresight angle
        psi = 2 * math.pi * d / (340 / 21000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (1 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (1 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude = math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 22000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 20000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N


        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 24000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 18000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 26000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 16000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 29000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        im=0
        re=0
        psi = 2 * math.pi * d / (340 / 13000) * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += (0.5 / 5) * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += (0.5 / 5) * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude += math.sqrt(re**2 + im**2)/N

        if logScale:
            magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

def get_dvar_pattern(N, d, wavelength, phi, amplitude_law, minimum_amplitude, dvarType, dvarHop, logScale=True):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    
    if dvarType == "di_zeng":
        D = [d + (i - 1) * dvarHop for i in range(N)]
    elif dvarType == "di_jian":
        D = [d + (N - i - 1) * dvarHop for i in range(N)]
    elif dvarType == "zeng_jian":
        D = [d + (N / 2 - 1 - abs(i - N / 2)) * dvarHop for i in range(N)]
    elif dvarType == "jian_zeng":
        D = [d + abs(i - N / 2) * dvarHop for i in range (N)]
    D[0] = 0
    
    phase_law = []
    phase = 0 
    for n in range(N):
        # 2pi * n * d / lambda  
        phase += -2 * math.pi * D[n] / wavelength * math.sin(math.radians(phi))
        phase_law.append(phase)

    theta = np.arange(-90,90,0.1);
    mag = []
    for i in theta:
        im=0
        re=0
        # Phase shift due to off-boresight angle
        n_psi = 0
        # Compute sum of effects of elements
        for n in range(N):
            n_psi += 2 * math.pi * D[n] / wavelength * math.sin(math.radians(i))
            im += amp_law[n] * math.sin(n_psi + phase_law[n])
            re += amp_law[n] * math.cos(n_psi + phase_law[n])
        magnitude = math.sqrt(re**2 + im**2)/N
        if logScale:
            magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

def get_measured_pattern(wav_path, c, sr, N, d, wavelength, f_c, phi, amplitude_law, minimum_amplitude, logScale=True):
    y, sr = librosa.load(wav_path, sr=sr)
    n_fft = 4800
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft)), ref=np.max)
    angle_list = [i for i in range(-60, 65, 5)]
    
    dB_list = []
    D_21k = D[int(f_c / 10)]
    len_D_time = len(D_21k)

    hop = (len_D_time - 1) / len(angle_list)
    for i, angle in enumerate(angle_list):
        D_left_index = int(i * hop + 1)
        D_right_index = int((i + 1) * hop)
        dB_sum = 0
        for j in range(D_left_index, D_right_index):
            dB_sum += D_21k[j]
        dB_list.append(dB_sum / (hop - 1))
    plt.plot(angle_list, dB_list, label = "Measured")
    plt.legend()
    plt.title("Expected Angle = "+str(phi))

    plt.show()

def get_rms_pattern(wav_path, c, sr, N, d, wavelength, f_c, phi, amplitude_law, minimum_amplitude, logScale=True):
    y, sr = librosa.load(wav_path, sr=sr)
    hop_length = 256
    frame_length = 512

    hop = sr * 8

    rmses = []
    for i in range(0, len(y), hop):
        x=y[i:i+hop]
        energy = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)
        ])
        # rms = librosa.feature.rms(y=y[i:i+hop], frame_length=frame_length, hop_length=hop_length, center=True)
        rmses.append(np.max(energy))

    angle_list = [i for i in range(-50, 60, 10)]
    
    
    plt.plot(angle_list, rmses, label = "Measured")
    plt.legend()
    plt.title("TDM, Expected Angle = -30 & 30")

    # fig_dir = os.path.join("figs", os.path.split(wav_path)[0], str(f_c))
    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)
    # plt.savefig(os.path.join(fig_dir, str(phi) + ".png"))
    plt.show()

# Compute antenna pattern
def get_wideband_pattern(N, d, wavelength, f_c, phi, amplitude_law, minimum_amplitude, logScale=True):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude)
    bins_num = 8
    bandwidth = 16000
    f_m_list, phase_law = get_wideband_phase_law(N, d, wavelength, phi, f_c, bandwidth=bandwidth, bins_num=bins_num)
    
    theta = np.arange(-90,90,0.1);
    mag = []
    for i in theta:
        im = 0
        re = 0
        
        # Compute sum of effects of elements
        for bin in range(bins_num):
            now_wavelength = 344 / f_m_list[bin]
            psi = 2 * math.pi * d / now_wavelength * math.sin(math.radians(i))
            for n in range(N):           
                im += amp_law[n] * math.sin(n*psi + phase_law[bin][n]) / bins_num
                re += amp_law[n] * math.cos(n*psi + phase_law[bin][n]) / bins_num
        magnitude = math.sqrt(re**2 + im**2)/N
        if logScale:
            magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

def get_wideband_phaseshift_wav(wav_path, c, sr, N, d, f_c, phi, f_c_left, f_c_right):
    dir_name = wav_path[:-4]+"_phaseshifted" + "/" + str(d) + "/" + str(phi)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    y, fs = librosa.load(wav_path, sr=None)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
 
    y_index_left = int((f_c_left) / hop)
    y_index_right = int((f_c_right) / hop)
    for n in range(0, N):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in range(y_index_left, y_index_right):
            f = i * hop
            now_wavelength = c / f
            now_phase_law = -2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi))
            fft_y[i] *= np.exp(1j * now_phase_law)
            fft_y[len_fft_y - i] /= np.exp(1j * now_phase_law)
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path.split("/")[-1]), y_hat, sr)

def get_two_wideband_phaseshift_wav(wav_path, c, sr, N, d, f_c, phi1, phi2):
    dir_name = str(N) + "/" + str(d)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    y, fs = soundfile.read(wav_path)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)
    print(hop, len(fft_y))
 
    y_index_left = int((f_c - 4e3) / hop)
    y_band_mid = int((f_c) / hop)
    y_index_right = int((f_c + 4e3) / hop)
    for n in tqdm(range(0, N)):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in tqdm(range(y_index_left, y_band_mid)):
            f = i * hop
            now_wavelength = c / f
            now_phase_law = -2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi1))
            fft_y[i] *= np.exp(1j * now_phase_law) 
            fft_y[len_fft_y - i] /= np.exp(1j * now_phase_law)
        for i in tqdm(range(y_band_mid, y_index_right)):
            f = i * hop
            now_wavelength = c / f
            now_phase_law = -2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi2))
            fft_y[i] *= np.exp(1j * now_phase_law) 
            fft_y[len_fft_y - i] /= np.exp(1j * now_phase_law)
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path), y_hat, sr)


def get_singleband_phaseshift_wav(wav_path, c, sr, N, d, wavelength, f_c, phi, amplitude_law, minimum_amplitude, logScale=True):
    # amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    dir_name = str(N) + "/" + str(d) + "/" + str(phi)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    y, fs = soundfile.read(wav_path)

    for n in tqdm(range(N)):
        fft_y = fft(y)
        now_wavelength = c / f_c
        now_phase_law = -2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi))
        fft_y[1:int(len(fft_y) / 2)] *= np.exp(1j * now_phase_law)
        fft_y[int(len(fft_y) / 2):] /= np.exp(1j * now_phase_law)
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path), y_hat, sr)



def get_phis_pattern(N, d, wavelength, phis, amplitude_law, minimum_amplitude, logScale=True):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude);
    phi_num = len(phis)
    amp_law = np.array(amp_law) * 0.15

    theta = np.arange(-90,90,0.1);
    mag = []
    for i in theta:
        im=0
        re=0
        psi = 2 * math.pi * d / wavelength * math.sin(math.radians(i))
        # Phase shift due to off-boresight angle
        for phi in phis:
            phase_law = get_phase_law(N, d, wavelength, phi);
            # Compute sum of effects of elements
            for n in range(N):
                im += amp_law[n] * math.sin(n*psi + phase_law[n])
                re += amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude = math.sqrt(re**2 + im**2)/N
        # if logScale:
        #     magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

def get_optimized_singleband_phaseshift_wav(wav_path, parameters_path, f_c, c, sr, angle_interval = 5):
    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])
    print(d_laws.shape, amp_laws.shape, phase_laws.shape)

    angle_num = amp_laws.shape[0]
    sweep_angle_begin = - angle_interval * int((angle_num - 1) / 2)
    sweep_angle_end = - sweep_angle_begin
    sweep_angles = np.arange(sweep_angle_begin, sweep_angle_end + 1, angle_interval)

    N = amp_laws.shape[1]
    print(sweep_angle_begin, sweep_angle_end, N)
    ### 读取音频
    y, fs = soundfile.read(wav_path)
    dir_name = "21k_single_band_optimized_d0016"
    for phi_index in range(0, angle_num + 1, 2):
        phi_dir_name = os.path.join(dir_name, str(sweep_angles[phi_index]))
        if not os.path.exists(phi_dir_name):
            os.makedirs(phi_dir_name)
        for n in range(N):
            fft_y = fft(y)
            amp_law = amp_laws[phi_index][n]
            phase_law = phase_laws[phi_index][n]
            fft_y[1:int(len(fft_y) / 2)] *= (amp_law * np.exp(1j * phase_law))
            fft_y[int(len(fft_y) / 2):] *= (amp_law / np.exp(1j * phase_law))
            y_hat = ifft(fft_y).flatten().astype(y.dtype)
            soundfile.write(os.path.join(phi_dir_name, str(n) + "-" + wav_path), y_hat, sr)

def get_optimized_wideband_phaseshift_wav(wav_path, parameters_path, N, d, f_c, c, sr, phi, angle_interval = 5, freq_interval = 10, sweep_angle_begin = -40):
    dir_name = wav_path[:-4]+"_optphaseshifted" + "/" + str(d) + "/" + str(phi)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])

    N = amp_laws.shape[2]

    freq_num = amp_laws.shape[0]
    freq_begin = int(f_c - freq_interval * (freq_num - 1) / 2)
    freq_end = int(f_c + freq_interval * (freq_num - 1) / 2)
    freqs = np.arange(freq_begin, freq_end + 1, freq_interval)

    y, fs = librosa.load(wav_path,sr = sr)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)

    phi_index = int((phi - sweep_angle_begin) / angle_interval)
    for n in range(N):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in freqs: 
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            amp_law = amp_laws[parameters_index][phi_index][n]
            phase_law = phase_laws[parameters_index][phi_index][n] - phase_laws[parameters_index][phi_index][0]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path.split("/")[-1]), y_hat, sr)

def get_optimized_widenullsteering_wideband_phaseshift_wav(wav_path, parameters_path, N, d, f_c, c, sr, freq_interval = 10, is_phasefluc = False):
    dir_name = wav_path[:-4]+"_optnullsteeringphaseshifted" + "/" + str(d) + "/" + "_".join(parameters_path.split("/")[0].split("_")[2:6])
    if is_phasefluc:
        dir_name = wav_path[:-4]+"_optnullsteeringphaseshiftedphasefluc" + "/" + str(d) + "/" + "_".join(parameters_path.split("/")[0].split("_")[2:6])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])

    N = amp_laws.shape[1]

    freq_num = amp_laws.shape[0]
    freq_begin = f_c - freq_interval * int((freq_num - 1) / 2)
    freq_end = f_c + freq_interval * int((freq_num - 1) / 2)

    y, fs = soundfile.read(wav_path)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)

    for n in range(N):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in range(freq_begin, freq_end + 1, freq_interval): 
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            amp_law = amp_laws[parameters_index][n]
            phase_law = phase_laws[parameters_index][n]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path.split("/")[-1]), y_hat, sr)

def get_micro_optimized_widenullsteering_wideband_phaseshift_wav(wav_path, parameters_path, N, d, f_c, c, sr, freq_interval = 10, is_phasefluc = False):
    dir_name = wav_path[:-4]+"_optnullsteeringphaseshifted" + "/" + str(d) + "/" + parameters_path[:-16]
    if is_phasefluc:
        dir_name = wav_path[:-4]+"_optnullsteeringphaseshiftedphasefluc" + "/" + str(d) + "/" + "_".join(parameters_path.split("_")[2:6])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])

    N = amp_laws.shape[1]

    freq_num = amp_laws.shape[0]
    freq_begin = f_c - freq_interval * int((freq_num - 1) / 2)
    freq_end = f_c + freq_interval * int((freq_num - 1) / 2)

    y, fs = soundfile.read(wav_path)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)

    for n in range(N):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in range(freq_begin, freq_end + 1, freq_interval): 
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            amp_law = amp_laws[parameters_index][n]
            phase_law = phase_laws[parameters_index][n]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path.split("/")[-1]), y_hat, sr)

def get_micro_phaserobust_optimized_widenullsteering_wideband_phaseshift_wav(wav_path, parameters_path, N, d, f_c, c, sr, freq_interval = 10, is_phasefluc = False):
    dir_name = wav_path[:-4]+"_optnullsteeringphaseshifted" + "/" + str(d) + "/" + parameters_path[:-16]
    if is_phasefluc:
        dir_name = wav_path[:-4]+"_optnullsteeringphaseshiftedphasefluc" + "/" + str(d) + "/" + parameters_path[:-16]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])

    N = amp_laws.shape[1]

    freq_num = amp_laws.shape[0]
    freq_begin = f_c
    freq_end = f_c + freq_interval * int(freq_num - 1)

    y, fs = soundfile.read(wav_path)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)

    for n in range(N):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in range(freq_begin, freq_end + 1, freq_interval): 
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            amp_law = amp_laws[parameters_index][n]
            phase_law = phase_laws[parameters_index][n]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path.split("/")[-1]), y_hat, sr)


def get_optimized_two_wideband_phaseshift_wav(wav_path, parameters_path, f_c, c, sr, phi1, phi2, angle_interval = 5, freq_interval = 10):
    ### 读取参数文件
    parameters_dict = load_json_file(parameters_path)
    d_laws = np.array(parameters_dict["d_law"])
    amp_laws = np.array(parameters_dict["amp_law"])
    phase_laws = np.array(parameters_dict["phase_law"])

    angle_num = amp_laws.shape[1]
    sweep_angle_begin = - angle_interval * int((angle_num - 1) / 2)
    sweep_angle_end = - sweep_angle_begin
    sweep_angles = np.arange(sweep_angle_begin, sweep_angle_end + 1, angle_interval)

    N = amp_laws.shape[2]

    freq_num = amp_laws.shape[0]
    freq_begin = f_c - freq_interval * int((freq_num - 1) / 2)
    freq_end = f_c + freq_interval * int((freq_num - 1) / 2)
    freqs = np.arange(freq_begin, freq_end + 1, freq_interval)

    y, fs = soundfile.read(wav_path)
    assert fs == sr
    y_len = len(y)
    hop = sr / y_len
    fft_y = fft(y)
    print(hop, len(fft_y))

    dir_name = "merge_a1_a2_optimized"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for n in tqdm(range(N)):
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        for i in range(freq_begin, f_c, freq_interval):
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            phi_index = int((phi1 - sweep_angle_begin) / angle_interval)
            amp_law = amp_laws[parameters_index][phi_index][n]
            phase_law = phase_laws[parameters_index][phi_index][n]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        for i in range(f_c, freq_end + 1, freq_interval):
            i_left_fftindex = int((i - freq_interval / 2) / hop)
            i_right_fftindex = int((i + freq_interval / 2) / hop)
            parameters_index = int((i - freq_begin) / freq_interval)
            phi_index = int((phi2 - sweep_angle_begin) / angle_interval)
            amp_law = amp_laws[parameters_index][phi_index][n]
            phase_law = phase_laws[parameters_index][phi_index][n]
            fft_y[i_left_fftindex:i_right_fftindex] *= (amp_law * np.exp(1j * phase_law))
            fft_y[len_fft_y - i_right_fftindex:len_fft_y - i_left_fftindex] *= (amp_law / np.exp(1j * phase_law))
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        soundfile.write(os.path.join(dir_name, str(n) + "-" + wav_path), y_hat, sr)

def restore_optimized_wideband_phaseshift_wav(wav_paths, output_path, c, sr, N, d, phi, f_c_left, f_c_right): 
    y_total = None
    for n in tqdm(range(0, N)):
        y, fs = soundfile.read(wav_paths[n])
        assert fs == sr
        y_len = len(y)
        hop = sr / y_len
        fft_y = fft(y)
        len_fft_y = len(fft_y)
        y_index_left = int((f_c_left) / hop)
        y_index_right = int((f_c_right) / hop)
        for i in tqdm(range(y_index_left, y_index_right)):
            f = i * hop
            now_wavelength = c / f
            now_phase_law = 2 * math.pi * n * d / now_wavelength * math.sin(math.radians(phi))
            fft_y[i] *= np.exp(1j * now_phase_law) 
            fft_y[len_fft_y - i] /= np.exp(1j * now_phase_law)
        y_hat = ifft(fft_y).flatten().astype(y.dtype)
        if y_total is None:
            y_total = copy.deepcopy(y_hat)
        else:
            y_total += y_hat
    y_total /= N
    soundfile.write(output_path, y_total, sr)
    # librosa.output.write_wav(output_path, y_total, sr)

def get_sdm_plusdirect_wav(dir_name, num_beam, wav_dirs, phis, N, d, sr):
    wavs_paths = []
    for i in range(num_beam):
        wav_dir = os.path.join(wav_dirs[i], str(d), str(phis[i]))
        wavs = sorted(os.listdir(wav_dir),key = lambda i:int(i[0]))
        wav_paths = [os.path.join(wav_dir, wav) for wav in wavs]
        wavs_paths.append(wav_paths)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(N):
        y = None
        for j in range(num_beam):
            y_tmp, sr = soundfile.read(wavs_paths[j][i])
            if y is None:
                y = copy.deepcopy(y_tmp)
            else:
                y += y_tmp
        y /= num_beam
        soundfile.write(os.path.join(dir_name, str(i)+".wav"), y, sr)

def get_sdm_plus_wav_with8channelwavs(wav_paths, num_beam, out_path, sr):
    sum_sig = None
    for wav_path in wav_paths:
        sig, sr_now = librosa.load(wav_path,sr=None, mono=False)
        assert sr_now == sr
        if sum_sig is None:
            sum_sig = copy.deepcopy(sig)
        else:
            sum_sig += sig
    sum_sig /= num_beam
    print(sum_sig.shape, np.max(sum_sig))
    sum_sig /= np.max(np.abs(sum_sig))
    soundfile.write(out_path, sum_sig.T, sr)

def gen_8channel(dir_name, N = 8):
    wave_all = None
    wavs = sorted(os.listdir(dir_name),key = lambda i:int(i[0]))
    for i in range(N):
        filename = os.path.join(dir_name, wavs[i])
        wav, sr = librosa.load(filename, sr=None)
        if wave_all is None:
            wave_all = np.zeros((N, wav.shape[0]))
        wave_all[i] = wav
    soundfile.write(dir_name+"_"+str(N)+"channel.wav", wave_all.T, sr)

# gen_8channel("16kto24k_N_FMCW")
# exit()

def gen_8_same_channel(wav_path, N = 8):
    wave_all=[]
    for i in range(N):
        wave0=wv.open(wav_path,'rb')
        wave_all.append(wave0)
    params=wave_all[0].getparams()
    output_file = wv.open(wav_path.split(".")[0]+"_"+str(N)+"channel.wav", 'wb')
    output_file.setparams(params)
    output_file.setnchannels(N)

    for i in range(params[3]): # params[3] 是采样点总数
        data = b''
        for j in range(N):
            sample = wave_all[j].readframes(1)
            if sample:
                data += sample
        output_file.writeframes(data)
    output_file.close()

# gen_8_same_channel("woman4_ssbup_sr96k_10s_1channel.wav")
# exit()

def gen_8_only_channel(wav_path, out_dir, N = 8):
    wave, sr = librosa.load(wav_path, sr = None)
    for i in range(N):
        wave_all = np.zeros((N, len(wave)))
        wave_all[i] = wave
        soundfile.write(os.path.join(out_dir, wav_path.split(".")[0]+"_"+str(i+1)+"channel.wav"), wave_all.T, sr)

def gen_8_inter_channel(wav_path, out_path, N = 8):
    wave, sr = librosa.load(wav_path, sr = None)
    wave = np.append(wave, np.zeros(sr))
    len_wave = len(wave)
    wave_all_1_shape = len_wave * N
    wave_all = np.zeros((N, wave_all_1_shape))
    for i in range(N):
        wave_all[i][int(i * len_wave):int((i + 1) * len_wave)] = wave
    soundfile.write(out_path, wave_all.T, sr)

def gen_8_inter_channel_in8channel(wav_path, out_path, N = 8):
    wave, sr = librosa.load(wav_path, sr = None, mono = False)
    wave = np.concatenate([wave, np.zeros((N, sr))], axis = 1)
    len_wave = wave.shape[1]
    wave_all_1_shape = len_wave * N
    wave_all = np.zeros((N, wave_all_1_shape))
    for i in range(N):
        wave_all[i][int(i * len_wave):int((i + 1) * len_wave)] = wave[i]
    soundfile.write(out_path, wave_all.T, sr)

def modulate_MVDR_angle(input_path, outputpath, c, d, target_theta=0, null_theta=30, Iswrite=True):
    input_audio, sr = soundfile.read(input_path)
    output_audio = np.zeros((8, len(input_audio)))

    frequencies=[18000, 24000]
    idx1 = int(frequencies[0] * len(input_audio) / sr)
    idx2 = int(frequencies[1] * len(input_audio) / sr)


    input_spec=np.fft.rfft(input_audio)
    f=np.arange(int(len(input_audio)//2))*(sr/len(input_audio))

    n=np.arange(0, 8, 1).reshape([-1,1])

    w=np.zeros((8,len(input_spec)),dtype=np.complex_)
    print(w.shape)
    for j in range(idx1,idx2):
        select_f=f[j]
        wavelength=c/select_f

        phase_diff= np.exp(- 1.0j * 2 * np.pi * n * d / wavelength * np.sin(np.radians(target_theta)))
        null_vec= np.exp( -1.0j * 2 * np.pi * n * d / wavelength * np.sin(np.radians(null_theta)))

        Rx=np.mat(1000*np.dot(null_vec,np.transpose(np.conj(null_vec)))+1*np.eye(8))
        a_theta_0 = np.mat(phase_diff)

        w1=Rx.I*a_theta_0/(a_theta_0.H *Rx.I*a_theta_0).H
        w1=np.squeeze(w1)
        w1=w1/np.max(w1)
        # print(w1)
        w[:,j]=w1

    for i in range(8):
        output_spec=np.copy(input_spec)
        for j in range(idx1,idx2):
            select_f=f[j]
            wavelength=c/select_f
            # phase_diff= - 2 * np.pi * i * d / wavelength * np.sin(np.radians(target_theta))
            # print(w[i:j],w[i:j].shape)
            output_spec[j]= input_spec[j] * w[i,j]

        output_audio[i]=np.fft.irfft(output_spec)
        
    print(output_audio.shape)
    if Iswrite:
        soundfile.write(outputpath, output_audio.T, sr)

    return output_audio, sr

def read_rew_freqres(freq_res_dir = "frequency_responses", N = 8):
    n_amps = []
    for i in range(N):
        rew_file = os.path.join(freq_res_dir, str(i + 1) + ".txt")
        rew_begin = False
        amps = []
        with open(rew_file, "r", encoding = 'gb2312') as f:
            for line in f.readlines():
                if rew_begin:
                    line_elms = line.strip("\n").split()
                    amp = 10 ** (float(line_elms[1]) / 20)
                    amps.append(amp)
                if line == "* Freq(Hz) SPL(dB) Phase(degrees)\n":
                    rew_begin = True
        n_amps.append(amps)
    n_amps = np.array(n_amps)
    return n_amps

def read_rew_phaseres(freq_res_dir = "frequency_responses", N = 8):
    n_amps = []
    for i in range(N):
        rew_file = os.path.join(freq_res_dir, str(i + 1) + ".txt")
        rew_begin = False
        amps = []
        with open(rew_file, "r", encoding = 'gb2312') as f:
            for line in f.readlines():
                if rew_begin:
                    line_elms = line.strip("\n").split()
                    amp = float(line_elms[2])
                    amps.append(amp)
                if line == "* Freq(Hz) SPL(dB) Phase(degrees)\n":
                    rew_begin = True
        n_amps.append(amps)
    n_amps = np.array(n_amps)
    return n_amps

def extend_8_channel(wav_path, out_path, N = 8, out_sec = 300):
    wave, sr = librosa.load(wav_path, sr = None, mono=False)
    extended_wave = np.zeros((8, 300*sr))
    wave_sec = wave.shape[1] / sr
    for i in range(30):
        extended_wave[:, int(i * 10 * sr):int((i + 1) * 10 * sr)] = wave
    soundfile.write(out_path, extended_wave.T, samplerate=96000)

# wav_path = "500Hz1ssbup_0_1500Hz1ssbup_40_8channel_null.wav"
# out_path = wav_path[:-4] + "_5min.wav"
# extend_8_channel(wav_path, out_path)
# exit()

def main():
    # TODO allow arbitrary amp and phase laws as argss
    parser = argparse.ArgumentParser(description="Generates BF pattern.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n","--number-elements", type=int, default=8,
                        help="number of elements")
    parser.add_argument("-c","--wave-celerity", type=float, default=344,
                        help="celerity of the wave in m/s (3.00E8 for light in vaccum, 340 for sound in air)")
    parser.add_argument("-d","--elements-spacing", type=float, default=0.008,
                        help="spacing between the elements in m")
    parser.add_argument("-f","--frequency", type=float, default=21000,
                        help="waveform frequency in Hz")
    parser.add_argument("-a","--steering-angle", type=float, default=0,
                        help="beam steering angle in deg")
    parser.add_argument("--amplitude-law", choices=['constant', 'linear', 'log_linear', 'poly2', 'poly3', 'chebshev'],
                        default='constant', help="amplitude law type")
    parser.add_argument("--minimum-amplitude", type=float,
                        default=dBtoLinear(-5), help="minimum normalized amplitude of the law")
    parser.add_argument("--polar", action="store_true",
                        help="display beam in polar graph")
    parser.add_argument("--log-scale", action="store_false",
                        help="display pattern in logarithmic scale")
    parser.add_argument("-s","--save-output", action="store_true",
                        help="save the output pattern and data in current folder")
    parser.add_argument("--directions_phis", type=str, default="0,40")
    parser.add_argument("--directions_wavpaths", type=str, default="ref_audio/woman1.wav,ref_audio/man1.wav")
    args = parser.parse_args();
    
    # Check parameters
    if args.number_elements <= 0:
         raise parser.error('The number of elements must be a positive integer.')
    if args.wave_celerity <= 0:
         raise parser.error('The wave celerity must be positive.')
    if args.elements_spacing <= 0:
         raise parser.error('The elements spacing must be positive.')
    if args.frequency <= 0:
         raise parser.error('The frequency must be positive.')
    if args.steering_angle < -90 or args.steering_angle > 90:
         raise parser.error('The steering angle must be in interval [-90;90].')
    if args.minimum_amplitude < 0 or args.minimum_amplitude > 1:
         raise parser.error('The minimum amplitude must be in interval [0;1].')
         
    # Parameters
    N = args.number_elements; #Elements 扬声器数量
    c = args.wave_celerity; #m/s 声音在空气中的传播速度
    d = args.elements_spacing; #m 扬声器之间的距离
    f = args.frequency; #Hz 频率
    phi = args.steering_angle; #deg 转向角度
    polar = args.polar; #True=polar patter, False=cartesian pattern
    logScale = args.log_scale; #True=output in dB, False=linear output
    amplitude_law = args.amplitude_law; #Amplitude law type
    minimum_amplitude = args.minimum_amplitude; #normalized amplitude
    
    # MODIFIED 1
    output_file = 'ptt_n_' + str(N) + '_d_'+ str(d) + '_phi_'+ str(phi) + '_' + amplitude_law if args.save_output else None;
    wavelength = c/f; #m

    phis_str = args.directions_phis.split(",")
    phis = [int(phi) for phi in phis_str]

    wavpaths = args.directions_wavpaths.split(",")
    
    ### 音频调制
    fc = 21000
    sr = 96000
    # # soundfile.write("21k_sr96k_10s_amp05_1channel.wav", carrier, sr)
    
    # # 边带调制
    write_root = "modulated_audios"
    modu_mode = "dsb"
    for wavpath in wavpaths:
        write_id = wavpath.split("/")[1][:-4]
        path = "ref_audio/" + write_id + ".wav"
        wav, sr = librosa.load(path, sr=sr)
        t = np.arange(len(wav)) / sr
        plt.plot(np.abs(rfft(wav)))
        if modu_mode == "dsb":
            wav = np.cos(2 * math.pi * fc * t) * (wav + 1) / 2
        elif modu_mode == "ssbdown":
            wav = np.cos(2 * math.pi * fc * t) * (wav + 1) / 2  - fftp.hilbert(wav) * np.sin(2 * math.pi * fc * t) / 2
        elif modu_mode == "ssbup":
            wav = np.cos(2 * math.pi * fc * t) * (wav + 1) / 2  + fftp.hilbert(wav) * np.sin(2 * math.pi * fc * t) / 2
        write_id_dir = os.path.join(write_root, write_id)
        if not os.path.exists(write_id_dir):
            os.makedirs(write_id_dir)
        write_path = os.path.join(write_id_dir, write_id + "_" + modu_mode + "_sr96k_10s_1channel.wav")
        soundfile.write(write_path, wav, sr)

    ## 获得宽频音频
    f = 21000
    sr = 96000
    f_c_left = f - 4e3
    f_c_right = f + 4e3
    # types = ["man", "woman"]
    # ids = np.arange(4, 5, 1)
    modu_modes = ["dsb"]
    for wavpath in wavpaths:
        type_id = wavpath.split("/")[1][:-4]
        for modu_mode in modu_modes:
            wav_path = os.path.join("modulated_audios", type_id, type_id+"_"+modu_mode+"_sr96k_10s_1channel.wav")
            for N in [8]:
                for d in [0.008]:
                    for phi in tqdm(phis):
                        get_wideband_phaseshift_wav(wav_path, c, sr, N, d, f, phi, f_c_left, f_c_right)
                        gen_8channel(wav_path[:-4]+"_phaseshifted" + "/" + str(d) + "/" + str(phi), N = 8)
    
    direct_wav_paths = []
    for i, wavpath in enumerate(wavpaths):
        type_id = wavpath.split("/")[1][:-4]
        direct_wav_paths.append(os.path.join("modulated_audios", type_id, type_id+"_"+modu_mode+"_sr96k_10s_1channel_phaseshifted", str(d), phis_str[i]+"_8channel.wav"))
    out_path = "".join(direct_wav_paths[0].split("/")[-3].split("_")[:2]) + "_"+phis_str[0]+"_" + "".join(direct_wav_paths[1].split("/")[-3].split("_")[:2]) + "_"+phis_str[1]+"_8channel_phaseshifted_d_" + str(d) + ".wav"
    out_path = os.path.join("modulated_audios_standard", out_path)
    get_sdm_plus_wav_with8channelwavs(direct_wav_paths, len(direct_wav_paths), out_path, sr)
    
    ### 优化多频音频
    N = 8
    d = 0.008
    f = 21000
    sr = 96000
    freq_interval = 10
    is_phasefluc = False
    modu_modes = ["dsb"]
    parameters_paths = [str(N) + "speakers_widenullsteering_target_"+phis_str[0]+"_null_"+phis_str[1]+"/parameters.json",
                        str(N) + "speakers_widenullsteering_target_"+phis_str[1]+"_null_"+phis_str[0]+"/parameters.json",]
    for wavpath in wavpaths:
        type_id = wavpath.split("/")[1][:-4]
        for parameters_path in tqdm(parameters_paths):    
            for modu_mode in modu_modes:
                wav_path = os.path.join("modulated_audios", type_id, type_id+"_"+modu_mode+"_sr96k_10s_1channel.wav")
                get_optimized_widenullsteering_wideband_phaseshift_wav(wav_path, parameters_path, N, d, f, c, sr, freq_interval=freq_interval, is_phasefluc=is_phasefluc)
                gen_8channel(wav_path[:-4]+"_optnullsteeringphaseshifted" + "/" + str(d) + "/" + "_".join(parameters_path.split("/")[0].split("_")[2:6]), N = 8)

    direct_wav_paths = []
    for i, wavpath in enumerate(wavpaths):
        type_id = wavpath.split("/")[1][:-4]
        direct_wav_paths.append(os.path.join("modulated_audios", type_id, type_id+"_"+modu_mode+"_sr96k_10s_1channel_optnullsteeringphaseshifted", str(d), "target_"+phis_str[i]+"_null_"+phis_str[i-1]+"_8channel.wav"))
    out_path = "".join(direct_wav_paths[0].split("/")[-3].split("_")[:2]) + "_"+phis_str[0]+"_" + "".join(direct_wav_paths[1].split("/")[-3].split("_")[:2]) + "_"+phis_str[1]+"_8channel_optnullsteeringphaseshifted_d_" + str(d) + ".wav"
    out_path = os.path.join("modulated_audios_wn", out_path)
    get_sdm_plus_wav_with8channelwavs(direct_wav_paths, len(direct_wav_paths), out_path, sr)

if __name__ == '__main__':
    main()