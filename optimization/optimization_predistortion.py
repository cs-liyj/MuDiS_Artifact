import matplotlib.pyplot as plt
import numpy as np
import soundfile
import math
from scipy.fftpack import fft, ifft
from scipy import signal
import wave as wv
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.fft
from torch.nn.utils.rnn import pad_sequence
import wave
import soundfile
import librosa
import argparse
import os

# 计算在angle角度收到的音频 N * len_freqs
def gen_rec_steering_vecs(N, ds, sr, angle, len_audio, freqs_begin, freqs_end):
    idx_begin = int(freqs_begin / sr * len_audio)
    idx_end = int(freqs_end / sr * len_audio)
    rec_steering_vecs = np.zeros([N, int(len_audio / 2 + 1)], dtype=complex)
    for i in range(N):
        for j in range(idx_begin, idx_end):
            f = j / len_audio * sr
            wave_length = c / f
            phase_diff = np.exp(1j * 2 * np.pi * ds[i] / wave_length * np.sin(np.radians(angle)))
            rec_steering_vecs[i][j] = phase_diff
    return rec_steering_vecs

def model_forward(audio_fft, amp_law, phase_law, rec_steering_vecs, num_target, N, len_audio):
    ### to_opt_vecs cal by sr and sec
    to_opt_vecs = torch.cat((torch.ones(N, 180000), amp_law * torch.cos(phase_law) + 1j * amp_law * torch.sin(phase_law), torch.ones(N, 240001)), 1)
    to_opt_audio_fft = audio_fft * to_opt_vecs # 8路空分后音频频谱乘待优化beam weight
    rec_audio_fft = to_opt_audio_fft.unsqueeze(0).expand(num_target, N, int(len_audio / 2 + 1)) * rec_steering_vecs # 优化后音频乘到达方向的steeting vectors
    rec_audio = torch.sum(torch.fft.irfft(rec_audio_fft, dim = 2), dim = 1) # 得到到达方向音频并8路相加
    rec_audio_nonlinead_fft = torch.fft.rfft(rec_audio + rec_audio ** 2, dim = 1) # 模拟非线性过程
    return rec_audio_nonlinead_fft, to_opt_audio_fft

def loss_func(target_fft, rec_audio_nonlinead_fft, spec_need_index = 30000):
    # 比较非线性后音频与原始音频相似度 前100可能出现直流分量 spec_need_index指只取低频
    target_fft_max = torch.max(torch.abs(target_fft[:,100:spec_need_index]))
    rec_audio_nonlinead_fft_max = torch.max(torch.abs(rec_audio_nonlinead_fft[:,100:spec_need_index]))
    loss = torch.linalg.vector_norm(target_fft[:,100:spec_need_index] / target_fft_max - rec_audio_nonlinead_fft[:,100:spec_need_index] / rec_audio_nonlinead_fft_max, ord = 2)
    return loss

def optimize_input_audio(num_target, target_angles, target_wavs, syned_wav, N, ds, c, sr, freqs_begin = 18000, freqs_end = 24000):
    min_loss = torch.inf

    ### 获取syned_wav的频谱 N * (len_audio / 2 + 1)
    audio_signals, sr = librosa.load(syned_wav, sr=sr, mono=False)
    len_audio = audio_signals.shape[1]
    audio_fft = torch.fft.rfft(torch.tensor(audio_signals))
    ### 获取target_wavs频谱并拼接 num_target * (len_audio / 2 + 1)
    target_fft = torch.zeros([num_target, int(len_audio / 2 + 1)], dtype=torch.complex64)
    for i in range(num_target):
        target_wav, sr = librosa.load(target_wavs[i], sr=sr)
        target_fft[i] = torch.fft.rfft(torch.tensor(target_wav * 4)) ######### 空分 /2 8路*8 所以*4

    len_vec = int((freqs_end - freqs_begin) / sr * len_audio)
    ### 待优化的beam weight amp_law, phase_law N * len_vec
    amp_law = Variable(torch.ones((N, len_vec)), requires_grad = True)
    phase_law = Variable(torch.zeros((N, len_vec)), requires_grad = True)

    ### 获取rec_steering_vecs 模拟音频到达目标方向的steering vectors num_target * N * (len_audio / 2 + 1)
    rec_steering_vecs = np.zeros([num_target, N, int(len_audio / 2 + 1)], dtype=complex)
    for i in range(num_target):
        angle = target_angles[i]
        rec_steering_vecs[i] = gen_rec_steering_vecs(N, ds, sr, angle, len_audio, freqs_begin, freqs_end)
    rec_steering_vecs = torch.tensor(rec_steering_vecs, dtype = torch.complex64)

    optimizer = optim.Adam([amp_law, phase_law], lr=0.01)
    num_epochs = 500
    print("Start optimization...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        ### 限定各个变量的范围
        amp_law.data.clamp_(0, 1)
        phase_law.data.clamp_(-math.pi, math.pi)

        ### 模型forward
        rec_audio_nonlinead_fft, _ = model_forward(audio_fft, amp_law, phase_law, rec_steering_vecs, num_target, N, len_audio)

        ### 计算损失函数
        loss = loss_func(target_fft, rec_audio_nonlinead_fft)

        ## 保存loss最小时的最优参数
        if loss < min_loss:
            min_loss = loss
            amp_law_optimized = amp_law.clone()
            phase_law_optimized = phase_law.clone()
        
        if loss < 0.001:            
            break

        ## 更新图
        loss.backward(retain_graph = True)
        optimizer.step()
        if epoch % 10 == 0:
            print('loss:{:.3f}'.format(loss.data))
    
    print("End optimization...")
    ### 获得当前最优解
    rec_audio_nonlinead_fft_opt, to_opt_audio_fft_opt = model_forward(audio_fft, amp_law_optimized, phase_law_optimized, rec_steering_vecs, num_target, N, len_audio)
    loss_opt = loss_func(target_fft, rec_audio_nonlinead_fft_opt)
    print('loss_opt:{:.3f}'.format(loss_opt.data))
    
    ###获取优化后音频
    opted_audio = torch.fft.irfft(to_opt_audio_fft_opt, dim = 1).detach().numpy()
    opted_audio /= np.max(np.abs(opted_audio))
    outpath = os.path.join("modulated_audios_disopt", syned_wav.split("/")[1][:-4]+"_distortionopted.wav")
    soundfile.write(outpath, opted_audio.T, samplerate=sr)


if __name__ == "__main__":

    # TODO allow arbitrary amp and phase laws as argss
    parser = argparse.ArgumentParser(description="Generates BF pattern.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--directions_phis", type=str, default="0,40")
    parser.add_argument("--directions_wavpaths", type=str, default="ref_audio/woman1.wav,ref_audio/man1.wav")
    parser.add_argument("--syned_wavpath", type=str, default="woman1ssbup_0_man1ssbup_40_8channel_optnullsteeringphaseshifted_d_0.008.wav")
    args = parser.parse_args()
    N = 8
    d = 0.008
    ds = np.array([d * i for i in range(N)])
    c = 344
    sr = 96000
    freqs_begin = 18000
    freqs_end = 24000

    phis_str = args.directions_phis.split(",")
    target_angles = [int(phi) for phi in phis_str]
    num_target = len(target_angles) ## 目标数量

    target_wavs = args.directions_wavpaths.split(",")
    syned_wav = args.syned_wavpath

    optimize_input_audio(num_target, target_angles, target_wavs, syned_wav, N, ds, c, sr, freqs_begin, freqs_end)
