

# how words get mapped
1. If assistnat is speaking it speaks 2 seconds of audio +- few words. TTS is run and the length of the audio is the length of the time stamp window. If a punctuation is detected at the end, then the minium length is set to 2 seconds. with white noise audio appended. Throw a warning log if audio is more then 4 seconds or less then 1 sec with the target text  of that audio.
2. if no assisstnat message is spoken then default length is 2 seconds
3. user message is mapped according the the ASR time alighnment. The System should process last 10 audio blocks and realignt last 5 audio block of user message for any correction.