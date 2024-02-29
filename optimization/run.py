import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--angles", type=str, default="0,40")
parser.add_argument("--audio_paths", type=str, default="ref_audio/woman1.wav,ref_audio/man1.wav")
args = parser.parse_args()

target_angles = args.angles
audio_paths = args.audio_paths


target_angles_str_list = target_angles.split(",")

os.system(" ".join(["python",
                    "optimization_difffreq_widenullsteering.py",
                    "--target_angle",
                    target_angles_str_list[0],
                    "--null_angle",
                    target_angles_str_list[1]]))

os.system(" ".join(["python",
                    "optimization_difffreq_widenullsteering.py",
                    "--target_angle",
                    target_angles_str_list[1],
                    "--null_angle",
                    target_angles_str_list[0]]))

os.system(" ".join(["python", 
                    "beamforming_pattern_gen.py", 
                    "--directions_phis", 
                    target_angles, 
                    "--directions_wavpaths",
                    audio_paths]))

audio_paths_strlist = audio_paths.split(",")
target_angles_strlist = target_angles.split(",")
os.system(" ".join(["python", 
                    "optimization_predistortion.py", 
                    "--directions_phis", 
                    target_angles, 
                    "--directions_wavpaths",
                    audio_paths,
                    "--syned_wavpath",
                    os.path.join("modulated_audios_wn",
                                "_".join([audio_paths_strlist[0].split("/")[1][:-4]+"dsb",
                                        target_angles_strlist[0],
                                        audio_paths_strlist[1].split("/")[1][:-4]+"dsb",
                                        target_angles_strlist[1],
                                        "8channel",
                                        "optnullsteeringphaseshifted",
                                        "d",
                                        "0.008.wav"]))]))

os.system(" ".join(["python", 
                    "simu_beamforming.py", 
                    "--channel8_wavpath",
                    "modulated_audios_disopt/woman1dsb_0_man1dsb_40_8channel_optnullsteeringphaseshifted_d_0.008_distortionopted.wav",
                    "--directions_phis",
                    target_angles]))

