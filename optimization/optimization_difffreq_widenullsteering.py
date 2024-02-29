#!/usr/bin/env python3

import wave
from cProfile import label
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg') #必须要写在这两个import中间
import matplotlib.pyplot as plt
import argparse
from scipy.fftpack import fft, ifft
# import librosa
import soundfile
from tqdm import tqdm
import copy
import os
import seaborn as sns
import pandas as pd

import torch
import torch.optim as optim
from torch.autograd import Variable

from scipy.signal import find_peaks

import json

import argparse

def write_json_file(json_dict, json_file_name):
    json_dumpted = json.dumps(json_dict, indent = 4)
    f = open(json_file_name, 'w')
    f.write(json_dumpted)
    f.close()

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
    
    chebshev_win = [0.5799,0.6603,0.8751,1.0000,1.0000,0.8751,0.6603,0.5799]

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

### 打印梯度 for debug
grads = {} # 存储节点名称与节点的grad
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def model_forward(N, d_law, amp_law, phase_law, sin_radians_matrix_thetas, wavelengths, len_freqs, len_matrix_thetas, cone_amp, target_angle_mask, nulls_mask, other_angle_mask):
    d_law_completed = torch.cat([d_law, torch.flip(d_law[:-1].unsqueeze(1),[0]).squeeze(1)])
    d_s = torch.cat([torch.zeros(1), torch.sum(torch.tril((d_law_completed/100).unsqueeze(0).expand(N-1,N-1)), dim=1)])

    wavelengths_expanded = wavelengths.unsqueeze(1).unsqueeze(2).expand(len_freqs, len_matrix_thetas, N)
    sin_radians_matrix_thetas_expanded = sin_radians_matrix_thetas.unsqueeze(0).unsqueeze(2).expand(len_freqs, len_matrix_thetas, N)
    d_s_expanded = d_s.unsqueeze(0).unsqueeze(1).expand(len_freqs, len_matrix_thetas, N)

    phase_delay = 2 * math.pi * sin_radians_matrix_thetas_expanded * d_s_expanded / wavelengths_expanded
    phase_law_expanded = phase_law.unsqueeze(1).expand(len_freqs, len_matrix_thetas, N)
    thetas_phases = phase_delay + phase_law_expanded

    cone_amp_expanded = cone_amp.unsqueeze(0).unsqueeze(2).expand(len_freqs, len_matrix_thetas, N)
    amp_law_expanded = amp_law.unsqueeze(1).expand(len_freqs, len_matrix_thetas, N)
    theta_amps = cone_amp_expanded * amp_law_expanded
    
    im_matrix = theta_amps * torch.sin(thetas_phases)
    re_matrix = theta_amps * torch.cos(thetas_phases)
    
    magnitude_matrix = torch.sqrt(torch.sum(im_matrix,dim=2) ** 2 + torch.sum(re_matrix,dim=2) ** 2) / N
    min_main_power = torch.sum(magnitude_matrix * target_angle_mask)

    max_null_power = torch.sum(magnitude_matrix * nulls_mask)

    max_side_power = torch.sum(magnitude_matrix * other_angle_mask)

    freqs_var = torch.var(torch.sum(magnitude_matrix * target_angle_mask, dim=1), dim=0)
    return min_main_power, max_null_power, max_side_power, freqs_var, magnitude_matrix

def loss_func(min_main_power, max_null_power, max_side_power, freqs_var):

    return - min_main_power + 0.5 * max_side_power + 10 * max_null_power + 1000 * freqs_var

def getdB(input):
    return 20 * np.log10(input)

def optimize_cone(N, wavelengths, d_theo, target_angle, nulls_angle, target_angle_width, nulls_width, logScale=True):
    min_loss = torch.inf
    # -60 - 60 以5度为间隔优化 一共25个角度
    edge_theta = 90
    matrix_thetas = np.arange(-edge_theta, edge_theta + 1, 1)
    # print(matrix_thetas.shape)
    ### Convert matrix_thetas to Radians (len_matrix_thetas, )
    sin_radians_matrix_thetas = torch.tensor(np.sin(np.radians(matrix_thetas)))

    wavelengths = torch.tensor(wavelengths)
    # 角度个数
    len_matrix_thetas = len(matrix_thetas)
    # 频率个数
    len_freqs = len(wavelengths)

    # N:扬声器的数量

    amp_law = Variable(torch.ones((len_freqs, N)), requires_grad = True)

    phase_law = Variable(torch.randn((len_freqs, N)), requires_grad = True)

    d_law = np.zeros(int(N / 2)) + d_theo
    d_law = Variable(torch.tensor(d_law), requires_grad = True)

    # get 3 mask
    target_angle_mask = torch.zeros(len_matrix_thetas)
    target_angle_mask[target_angle + edge_theta - target_angle_width:target_angle + edge_theta + target_angle_width + 1] = 1
    target_angle_mask = target_angle_mask.unsqueeze(0).expand(len_freqs, len_matrix_thetas)

    nulls_mask = torch.zeros(len_matrix_thetas)
    nulls_mask[nulls_angle + edge_theta - nulls_width:nulls_angle + edge_theta + nulls_width + 1] = 1
    nulls_mask = nulls_mask.unsqueeze(0).expand(len_freqs, len_matrix_thetas)

    other_angle_mask = torch.ones((len_freqs, len_matrix_thetas)) - target_angle_mask - nulls_mask
    cone_amp = torch.ones(len_matrix_thetas)

    optimizer = optim.Adam([amp_law, phase_law], lr=0.01)
    
    ### epoch数量
    num_epochs = 2000
    print("Start optimization...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        ### 限定各个变量的范围
        amp_law.data.clamp_(0.1, 1.0)

        ### 模型forward
        min_main_power, max_null_power, max_side_power, freqs_var, _ = model_forward(N, d_law, amp_law, phase_law, sin_radians_matrix_thetas, wavelengths, len_freqs, len_matrix_thetas, cone_amp, target_angle_mask, nulls_mask, other_angle_mask)

        ### 计算损失函数
        loss = loss_func(min_main_power, max_null_power, max_side_power, freqs_var)

        ## 更新图
        loss.backward(retain_graph = True)

        ## 保存loss最小时的最优参数
        if loss < min_loss:
            min_loss = loss
            amp_law_optimized = amp_law.clone()
            phase_law_optimized = phase_law.clone()
            d_law_optimized = d_law.clone()
        
        optimizer.step()
        if epoch % 50 == 0:
            print('loss:{:.3f} min_main_power:{:.3f} max_null_power:{:.3f} max_side_power:{:.3f} freqs_var:{:.3f}'.format(loss.item(), min_main_power.item(), max_null_power.item(), max_side_power.item(), freqs_var.item()))
    
    print("End optimization...")
    ### 获得当前最优解
    min_main_power_opt, max_null_power_opt, max_side_power_opt, freqs_var_opt, magnitude_matrix_opt = model_forward(N, d_law_optimized, amp_law_optimized, phase_law_optimized, sin_radians_matrix_thetas, wavelengths, len_freqs, len_matrix_thetas, cone_amp, target_angle_mask, nulls_mask, other_angle_mask)
    loss_opt = loss_func(min_main_power_opt, max_null_power_opt, max_side_power_opt, freqs_var_opt)
    print('loss_opti:{:.3f} min_main_power_opti:{:.3f} max_null_power_opti:{:.3f} max_side_power_opti:{:.3f} freqs_var_opti:{:.3f}'.format(loss_opt.item(), min_main_power_opt.item(), max_null_power_opt.item(), max_side_power_opt.item(), freqs_var_opt.item()))
    # plt.plot(matrix_thetas, magnitude_matrix_opt[0].detach().numpy())
    ### 获得理论值
    d_law_theo = torch.zeros(int(N / 2)) + d_theo
    amp_law_theo = torch.ones((len_freqs, N))
    phase_law_theo = np.zeros((len_freqs, N))
    for i in range(len_freqs):
        phase_law_theo[i] = get_phase_law(N, d_theo / 100, wavelengths[i], target_angle)
    phase_law_theo = torch.tensor(phase_law_theo)
    min_main_power_theo, max_null_power_theo, max_side_power_theo, freqs_var_theo, magnitude_matrix_theo = model_forward(N, d_law_theo, amp_law_theo, phase_law_theo, sin_radians_matrix_thetas, wavelengths, len_freqs, len_matrix_thetas, cone_amp, target_angle_mask, nulls_mask, other_angle_mask)
    loss_theo = loss_func(min_main_power_theo, max_null_power_theo, max_side_power_theo, freqs_var_theo)
    print('loss_theo:{:.3f} min_main_power_theo:{:.3f} max_null_power_theo:{:.3f} max_side_power_theo:{:.3f} freqs_var_theo:{:.3f}'.format(loss_theo.item(), min_main_power_theo.item(), max_null_power_theo.item(), max_side_power_theo.item(), freqs_var_theo.item()))

    with_heat_map = False
    if with_heat_map:
        plt.figure(1)
        sns.set_theme()
        magnitude_matrix_df = pd.DataFrame(magnitude_matrix.detach().numpy(), index=matrix_thetas, columns=matrix_thetas)
        magnitude_matrix_theo_df = pd.DataFrame(magnitude_matrix_theo.detach().numpy(), index=matrix_thetas, columns=matrix_thetas)
        magnitude_matrix_diff_df = pd.DataFrame(magnitude_matrix_theo.detach().numpy() - magnitude_matrix.detach().numpy(), index=matrix_thetas, columns=matrix_thetas)
        sns.heatmap(magnitude_matrix_df)

        plt.figure(2)
        sns.heatmap(magnitude_matrix_theo_df)
        plt.show()

    with_magnitude_matrix_figures = False
    if with_magnitude_matrix_figures:
        magnitude_matrix = magnitude_matrix.detach().numpy()
        figs_dir = "optimization_results"
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        for i, theta in enumerate(matrix_thetas):
            plt.clf()
            plt.plot(matrix_thetas, magnitude_matrix[i])
            plt.ylim(top=1)
            plt.xlabel("Phi")
            plt.ylabel("Magnitude")
            plt.title("Target Angle = " + str(theta))
            plt.savefig(os.path.join(figs_dir, str(theta)+".png"))
    
    amp_law_optimized = amp_law_optimized.detach().numpy()
    phase_law_optimized = phase_law_optimized.detach().numpy()
    d_law_optimized = d_law_optimized.detach().numpy()
    d_law_optimized = np.concatenate((d_law_optimized,d_law_optimized[:-1][::-1]))
    return matrix_thetas, d_law_optimized, amp_law_optimized, phase_law_optimized

def get_optimized_pattern(N, wavelengths, matrix_thetas, d_law, amp_law, phase_law, logScale=False):
    thetas = np.arange(-90,91,1)
    # cone_amp = - thetas ** 2 / (90 ** 2) + 1
    cone_amp = np.ones(len(thetas))
    wavelengths_mags = []
    for lambda_index, wavelength in enumerate(wavelengths):
        mag = []
        for theta in thetas:
            im = 0
            re = 0
            n_d_s = 0
            for n in range(N):
                n_d_s += d_law[n - 1]/100 if n != 0 else 0
                theta_phase_delay = 2 * math.pi * n_d_s / wavelength * math.sin(math.radians(theta))
                im += cone_amp[theta+90] * amp_law[lambda_index][n] * math.sin(theta_phase_delay + phase_law[lambda_index][n])
                re += cone_amp[theta+90] * amp_law[lambda_index][n] * math.cos(theta_phase_delay + phase_law[lambda_index][n])
                # im += cone_amp[theta+90] * 1 * math.sin(theta_phase_delay + phase_law[lambda_index][i][n])
                # re += cone_amp[theta+90] * 1 * math.cos(theta_phase_delay + phase_law[lambda_index][i][n])
            magnitude = math.sqrt(re**2 + im**2)/N
            if logScale:
                magnitude = 20*math.log10(magnitude)
            mag.append(magnitude)
        wavelengths_mags.append(mag)
    return wavelengths_mags

# Compute antenna pattern
def get_pattern(N, d, wavelength, phi, amplitude_law, minimum_amplitude, logScale=False):
    """Computes an array pattern given N (number of elements),
    d (spacing between elements in m), wavelength (in m),
    phi (beam steering angle), amplitude_law (type of law)
    and minAmp (minimum amplitude).
    """
    # Compute phase and amplitudes laws
    amp_law = get_amplitude_law(N, amplitude_law, minimum_amplitude)
    phase_law = get_phase_law(N, d, wavelength, phi)
    
    theta = np.arange(-90,91,1)
    # cone_amp = - theta ** 2 / (90 ** 2) + 1
    cone_amp = np.ones(len(theta))
    mag = []
    for i in theta:
        im=0
        re=0
        # Phase shift due to off-boresight angle
        psi = 2 * math.pi * d / wavelength * math.sin(math.radians(i))
        # Compute sum of effects of elements
        for n in range(N):
            im += cone_amp[i+90] * amp_law[n] * math.sin(n*psi + phase_law[n])
            re += cone_amp[i+90] * amp_law[n] * math.cos(n*psi + phase_law[n])
        magnitude = math.sqrt(re**2 + im**2)/N
        if logScale:
            magnitude = 20*math.log10(magnitude)
        mag.append(magnitude)
        
    return theta, mag, amp_law, phase_law

# def plot_pattern(expected_angle, theta, mag, amp_law, phase_law, polar=False, output_file=None, compare_type=None):
def plot_pattern(theta, mag, target_phi=None, output_file=None, compare_type=None):
    plt.grid(True)
    plt.xlabel("Phi")
    plt.ylabel("Magnitude")
    plt.xlim([-90, 90])
    plt.ylim(top=1)

    if compare_type == None:
        plt.plot(theta, mag, label="Theoretical", alpha=0.7)
    else:
        plt.plot(theta, mag, label=compare_type, alpha=0.7)
    
    # Show and save plot
    if output_file is not None and output_file != "show":
        plt.title("Expected Angle = " + str(target_phi) + "°")
        plt.legend(loc = "upper right")
        plt.savefig(output_file + '.png')
    elif output_file == "show":
        plt.show()


def main():
    # TODO allow arbitrary amp and phase laws as argss
    parser = argparse.ArgumentParser(description="Generates BF pattern.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n","--number-elements", type=int, default=8,
                        help="number of elements")
    parser.add_argument("-c","--wave-celerity", type=float, default=344,
                        help="celerity of the wave in m/s (3.00E8 for light in vaccum, 340 for sound in air)")
    parser.add_argument("-d","--elements-spacing", type=float, default=0.008, # Need Modifying
                        help="spacing between the elements in m")
    parser.add_argument("-f","--frequency", type=float, default=21000,
                        help="waveform frequency in Hz")
    parser.add_argument("-a","--steering-angle", type=float, default=30,
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
    parser.add_argument('--target_angle', type=int, default=0)
    parser.add_argument('--null_angle', type=int, default=40)
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
    
    wavelength = c/f; #m
    band_width = 3000
    band_spacing = 10
    wavelengths = [c / freq for freq in range(int(f - band_width), int(f + band_width + 1), int(band_spacing))]

    target_angle = args.target_angle
    nulls_angle = args.null_angle
    target_angle_width = 10
    nulls_width = 10
    
    matrix_thetas, d_law_optimized, amp_law_optimized, phase_law_optimized = optimize_cone(N, wavelengths, d * 100, target_angle, nulls_angle, target_angle_width, nulls_width) #  d * 100 (convert m to cm)

    wavelengths_mags = get_optimized_pattern(N, wavelengths, matrix_thetas, d_law_optimized, amp_law_optimized, phase_law_optimized)

    output_file_dir = str(N) + "speakers_widenullsteering_target_" + str(target_angle) + "_null_" + str(nulls_angle)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    f_num = phase_law_optimized.shape[0]
    f_phase_num = phase_law_optimized.shape[1]
    for i in range(f_num):
        base_phase = phase_law_optimized[i][0]
        for j in range(f_phase_num):
            phase_law_optimized[i][j] -= base_phase
    
    json_dict = {"d_law":np.float64(d_law_optimized).tolist(), "amp_law":np.float64(amp_law_optimized).tolist(), "phase_law":np.float64(phase_law_optimized).tolist()}
    

    write_json_file(json_dict, os.path.join(output_file_dir, "parameters.json"))

if __name__ == '__main__':

    main()
