Folder File Sturcture

Folder "metasurface" - show our printed cell structure and corresponding metasurface for 4*8 ultrasonic transudcer array

Folder "optimization"- show the codes about wide-nulling algorithm and distortiong reduction algorithm for 2 beams
	- "ref_audio":  The original audios for modulation.
	- “modulated_audios”: Some intermediate results, please ignore them. 
	- "modulated_audios_standard": 8-channel audios with standard amplitude modulation.
	- "modulated_audios_wn": 8-channel audios with wide-nulling optimization.
	- "modulated_audios_disopt": 8-channel audios with distortion reduction algorithm —— the final 8-channel audio
	- "simu_output": Simulation of the results for low frequency audible audio output

Folder "evaluation" - show the codes to evaluate the performance
	-  "ref_audio":  The original audios for modulation
	- "example": an example of the collected audios at different angles
	- "cal_metrics.py" : calculate the metrics about SNR, PESQ, MCD

