import pretty_midi
import soundfile as sf
import os

def midi_to_wav(midi_path, sample_rate=44100):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    audio_data = midi_data.synthesize(fs=sample_rate)

    # split .mid file name to get file name, append .wav for wave file, then create path '/output/wav/{file name}.wav'.
    wav_path = os.path.join('output', 'wav', os.path.splitext(os.path.basename(midi_path))[0] + '.wav')

    sf.write(wav_path, audio_data, sample_rate)

    return wav_path