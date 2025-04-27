import pretty_midi
import soundfile as sf
import numpy as np
import os

def midi_to_wav(midi_path, sample_rate=44100):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    audio_data = midi_data.synthesize(fs=sample_rate)

    # split .mid file name to get file name, append .wav for wave file, then create path '/output/wav/{file name}.wav'.
    wav_path = os.path.join('output', 'wav', os.path.splitext(os.path.basename(midi_path))[0] + '.wav')
    
    os.makedirs('output/wav', exist_ok=True)
    sf.write(wav_path, np.ravel(audio_data), sample_rate, format='WAV', subtype='PCM_16')

    return wav_path