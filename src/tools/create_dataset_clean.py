import os
import shutil
import sys

relative_path = str(sys.argv[1])
directory = os.path.join(str(os.getcwd()),relative_path)

with open(os.path.join(directory,'predictions.csv'), "r") as f:
    data = f.readlines()
    mask = [0] * len(data)
    y_list = []
    t_list = []

    for i, line in enumerate(data):
        words = line.split(',')
        t_ = int(float(words[1]))
        y_ = int(float(words[0]))
        y_list.append(y_)
        t_list.append(t_)

        y_conf_original = float(words[2+y_])
        y_conf_adv = float(words[12+y_])
        t_conf_original = float(words[2+t_])
        t_conf_adv = float(words[12+t_])

        if y_ != t_:
            mask[i] = 1



path_src = directory
path_dst = os.path.join(str(os.getcwd()),os.path.join("..","Datasets",""))


files = os.listdir(path_src)
x_names = [x for x in files if 'adv' not in x and 'noise' not in x and 'x' in x and 'wav' in x]
name_to_label = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9}
label_to_name = {v: k for k, v in name_to_label.items()}

targeted = False
if 'untargeted' in path_src:
    if 'RG' in path_src:
        path_dst = os.path.join(path_dst,"speech_eval_RG_untargeted_clean")
    elif 'LG' in path_src:
        path_dst = os.path.join(path_dst,"speech_eval_LG_untargeted_clean")
elif 'targeted' in path_src:
    if 'RG' in path_src:
        targeted = True
        path_dst = os.path.join(path_dst,"speech_eval_RG_targeted_clean")
    elif 'LG' in path_src:
        targeted = True
        path_dst = os.path.join(path_dst,"speech_eval_LG_targeted_clean")


if os.path.isdir(path_dst):
    input('path already exists')
    quit
else:
    os.mkdir(path_dst)


for name in x_names:
    
    i = int(name.strip('x*.wav'))
    if targeted:
        i += len(y_list) - 91
    else:
        i += len(y_list) - 100
    if targeted:
        if y_list[i] != t_list[i]:
            dst = os.path.join(path_dst,label_to_name[y_list[i]])
            if not os.path.isdir(dst):
                 os.mkdir(dst)
            dst = os.path.join(dst,name)
            
            src = os.path.join(path_src, name)
            shutil.copyfile(src=src, dst=dst)
    else:
        dst = os.path.join(path_dst,label_to_name[y_list[i]])
        if not os.path.isdir(dst):
                os.mkdir(dst)
        dst = os.path.join(dst,name)

        src = os.path.join(path_src, name)
        shutil.copyfile(src, dst=dst)

#import pdb; pdb.set_trace()