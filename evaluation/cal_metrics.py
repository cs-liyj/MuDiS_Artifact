import librosa
import pesq
from scipy import signal
import numpy as np
import argparse

def cal_pesq(ref_audio,test_audio,sr=8000):
    pesq_score = pesq.pesq(sr, ref_audio, test_audio, mode='nb')
    return pesq_score


def cal_SNR(ref_audio,test_audio):
    SNR = 10 * np.log10(np.sum(ref_audio ** 2) / np.sum((ref_audio - test_audio) ** 2))
    return SNR

def cal_MCD(ref_audio, test_audio, sr):
    mel1 = librosa.feature.melspectrogram(y=ref_audio, sr=sr)
    mel2 = librosa.feature.melspectrogram(y=test_audio, sr=sr)

    cepstral1 = librosa.power_to_db(mel1, ref=np.max)
    cepstral2 = librosa.power_to_db(mel2, ref=np.max)

    MCD = np.sqrt(np.mean((cepstral1 - cepstral2) ** 2))

    return MCD

def read_audio(audio_path,time):
    audio,sr=librosa.load(audio_path,sr=8000)
    audio=audio[:int(time * sr)]
    b,a = signal.butter(8, 0.3, 'lowpass')
    audio = signal.filtfilt(b,a,audio)
    audio = audio / np.max(audio)

    return audio,sr

def cal_metrics(ref_audio_path,test_audio_path,time):

    ref_audio,sr = read_audio(ref_audio_path,time)
    test_audio,sr = read_audio(test_audio_path,time)

    pesq_score = cal_pesq(ref_audio,test_audio,sr)
    SNR = cal_SNR(ref_audio,test_audio)
    MCD = cal_MCD(ref_audio,test_audio,sr)

    print("SNR: %.2f, PESQ: %.2f, MCD:%.2f" % (SNR,pesq_score,MCD))


if __name__=='__main__':

    #The test audio should be aligned in advance

    time = 10
    parser = argparse.ArgumentParser(description="Generates BF pattern.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_audio_path", type=str, default='./example/woman1dsb_0_man1dsb_40_opt_0.wav')
    parser.add_argument("--ref_audio_path", type=str, default='./ref_audio/woman1_8k.wav')
    args = parser.parse_args()

    test_audio_path = args.test_audio_path
    ref_audio_path = args.ref_audio_path


    print("ref_audio_path:",ref_audio_path)
    print("test_audio_path:",test_audio_path)

    # calculate the metrics
    cal_metrics(ref_audio_path,test_audio_path,time)


