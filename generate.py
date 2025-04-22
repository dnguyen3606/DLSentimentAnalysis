import pickle, os, torch, time, json
import numpy as np
from utils.models import TransformerModel
from utils.emopia import get_random_string, write_midi
from collections import OrderedDict

def generate(num_songs=1, emotion_tag=1):
    # path
    path_gendir = 'output/midi/'
    path_saved_ckpt = 'models/loss_25_params.pt'
    path_dictionary = 'data/emopia/co-representation/dictionary.pkl'

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []   # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    n_token = len(n_class)

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    
    try:
        net.load_state_dict(torch.load(path_saved_ckpt))
    except:
        state_dict = torch.load(path_saved_ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    sidx = 0
    while sidx < num_songs:
        # try:
        start_time = time.time()
        print('current idx:', sidx)

        if n_token == 8:
            path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format( str(emotion_tag), get_random_string(10)))        
            res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token)

        if res is None:
            continue
        # np.save(path_outfile + '.npy', res)
        write_midi(res, path_outfile + '.mid', word2event)
        print(f'music generated at {path_outfile}.mid')

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    return path_outfile

    # runtime_result = {
    #     'song_time':song_time_list,
    #     'words_len_list': words_len_list,
    #     'ave token time:': sum(words_len_list) / sum(song_time_list),
    #     'ave song time': float(np.mean(song_time_list)),
    # }

    # with open('runtime_stats.json', 'w') as f:
    #     json.dump(runtime_result, f)