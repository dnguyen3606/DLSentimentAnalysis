import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import random, string, os, zipfile, gdown, shutil

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

def write_midi(words, path_outfile, word2event):
    
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)
                
                all_notes.append(
                    Note(
                        pitch=int(pitch), 
                        start=cur_pos, 
                        end=end, 
                        velocity=int(velocity))
                    )
            except:
                continue
        else:
            pass
    
    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def download_and_extract(google_drive_id):
    if google_drive_id == '19Seq18b2JNzOamEQMG1uarKjj27HJkHu': # pretrained transformer case
        download_to_path = 'models/'
        expected_file = os.path.join(download_to_path, 'loss_25_params.pt')
    elif google_drive_id == '17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP': # co-representation/dictionary case
        download_to_path = 'data/emopia/co-representation/'
        expected_file = os.path.join(download_to_path, 'emopia_data.npz')
    else:
        return # dont use this function to download anything else. its unnecessary after all.
    
    if os.path.exists(expected_file):
        return # if file already exists skip all this nonsense

    zip_path = os.path.join(download_to_path, 'archive.zip')

    gdown.download(
        url=f'https://drive.google.com/uc?id={google_drive_id}', 
        output=zip_path, 
        quiet=False,
    )

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_to_path)

    flatten_directory(download_to_path)
    
    os.remove(zip_path)

def flatten_directory(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(root_dir,   filename)
            shutil.move(src, dst)  
        if dirpath != root_dir:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass 