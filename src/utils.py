import os, shutil, pickle
import torch, torchvision
import numpy as np 
import pydub
from skimage import io as skio
from skimage import transform as sktransform
from scipy.io.wavfile import read as wavread
import argparse
import os.path
import audio_utils 
import random

#####################
#   Dataset Utils   #
#####################
def create_speech_commands_dataset_atlas(directory, force=False):
    ''' http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz '''
    
    if not os.path.isfile(os.path.join(directory,'audio_atlas.pkl')):
        
        paths = {'train': list(), 'validation': list(), 'evaluation': list()}
        
        for class_name in os.listdir(directory):
            folder_name = os.path.join(directory, class_name)
            path_names = [os.path.join(folder_name,v) for v  in os.listdir(folder_name)]
            
            random.shuffle(path_names)

            paths['train'] += path_names[:int(len(path_names)*0.8)]
            
            paths['validation'] += path_names[int(len(path_names)*0.8):int(len(path_names)*0.95)]
            
            paths['evaluation'] += path_names[int(len(path_names)*0.95):]
        print(len(paths['train']))
        print(len(paths['validation']))
        print(len(paths['evaluation']))
        with open(os.path.join(directory,'audio_atlas.pkl'), 'wb') as file:
            pickle.dump(paths, file, pickle.HIGHEST_PROTOCOL)
        
    if not os.path.isfile(os.path.join(directory,'name_to_label.pkl')):
        name_to_label = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9}
        with open(os.path.join(directory,'name_to_label.pkl'), 'wb') as file:
            pickle.dump(name_to_label, file, pickle.HIGHEST_PROTOCOL)


def create_speech_commands_dataset_atlas_eval(directory, force=False):
    ''' http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz '''

    
    if not os.path.isfile(os.path.join(directory,'audio_atlas.pkl')):
        
        paths = {'train': list(), 'validation': list()}
        
        for class_name in os.listdir(directory):
            folder_name = os.path.join(directory, class_name)
            path_names = [os.path.join(folder_name,v) for v  in os.listdir(folder_name)]
            
            paths['train'] += path_names
            paths['validation'] += path_names
            
        with open(os.path.join(directory,'audio_atlas.pkl'), 'wb') as file:
            pickle.dump(paths, file, pickle.HIGHEST_PROTOCOL)
        
    if not os.path.isfile(os.path.join(directory,'name_to_label.pkl')):
        name_to_label = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9}
        with open(os.path.join(directory,'name_to_label.pkl'), 'wb') as file:
            pickle.dump(name_to_label, file, pickle.HIGHEST_PROTOCOL)


def create_FMA_small_dataset_atlas(directory, force=False):
    if not os.path.isfile(os.path.join(directory,'audio_atlas.pkl')):
        paths = {'train': list(), 'validation': list()}
        for class_name in os.listdir(directory):
            folder_name = os.path.join(directory, class_name)
            path_names = [os.path.join(folder_name,v) for v in os.listdir(folder_name)]

            paths['train'] += path_names[:int(len(path_names)*0.85)]
            paths['validation'] += path_names[int(len(path_names)*0.85):]

        with open(os.path.join(directory,'audio_atlas.pkl'), 'wb') as file:
            pickle.dump(paths, file, pickle.HIGHEST_PROTOCOL)
    
    if not os.path.isfile(os.path.join(directory,'name_to_label.pkl')):
        name_to_label = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}
        with open(os.path.join(directory,'name_to_label.pkl'), 'wb') as file:
            pickle.dump(name_to_label, file, pickle.HIGHEST_PROTOCOL)
        
def create_mnist_dataset_atlas(directory, force=False):
    if not os.path.isfile(os.path.join(directory,'mnist_atlas.pkl')):
        
        paths = {'train': list(), 'validation': list()}
        
        for class_name in os.listdir(directory):
            folder_name = os.path.join(directory, class_name)
            path_names = [os.path.join(folder_name,v) for v  in os.listdir(folder_name)]
            paths['train'] += path_names[:int(len(path_names)*0.85)]
            paths['validation'] += path_names[int(len(path_names)*0.85):]
            

        with open(os.path.join(directory,'mnist_atlas.pkl'), 'wb') as file:
            pickle.dump(paths, file, pickle.HIGHEST_PROTOCOL)
        
    if not os.path.isfile(os.path.join(directory,'name_to_label.pkl')):
        name_to_label = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
        with open(os.path.join(directory,'name_to_label.pkl'), 'wb') as file:
            pickle.dump(name_to_label, file, pickle.HIGHEST_PROTOCOL)
    
def create_dogs_cats_dataset_atlas(image_directory, force=False):

    ''' Downloaded from https://www.kaggle.com/lingjin525/dogs-and-cats-fastai '''

    if 'dogscats' in os.listdir(image_directory):
        shutil.rmtree(os.path.join(image_directory,'dogscats'))
    if 'sample' in os.listdir(image_directory):
        shutil.rmtree(os.path.join(image_directory,'sample'))
    if 'test1' in os.listdir(image_directory):
        os.rename(os.path.join(image_directory,'test1'),os.path.join(image_directory,'test'))
    
    if not os.path.isfile(os.path.join(image_directory,'images_atlas.pkl')):

        image_paths = {}
        image_paths['train'] = \
            [os.path.join(image_directory,'train','cats',v) for v in os.listdir(os.path.join(image_directory,'train','cats'))] \
            + [os.path.join(image_directory,'train','dogs',v) for v in os.listdir(os.path.join(image_directory,'train','dogs'))]
        image_paths['validation'] = \
            [os.path.join(image_directory,'valid','cats',v) for v in os.listdir(os.path.join(image_directory,'valid','cats'))] \
            + [os.path.join(image_directory,'valid','dogs',v) for v in os.listdir(os.path.join(image_directory,'valid','dogs'))]
        with open(os.path.join(image_directory,'images_atlas.pkl'), 'wb') as file:
            pickle.dump(image_paths, file, pickle.HIGHEST_PROTOCOL)

def create_imagenet_dataset_atlas(image_directory):

    ''' Downloaded using https://github.com/mf1024/ImageNet-Datasets-Downloader with command:
            $ python3 downloader.py -data_root ../Datasets/imagenet -use_class_list True \
                -class_list n01484850 n02007558 n07753592 n07745940 n02051845 n02129604 \
                -images_per_class 2000
    '''

    if not os.path.isfile(os.path.join(image_directory,'images_atlas.pkl')):

        image_paths = {'train': list(), 'validation': list()}
        for class_name in os.listdir(image_directory):
            folder_name = os.path.join(image_directory,class_name)
            path_names = [os.path.join(folder_name,v) for v  in os.listdir(folder_name)]
            image_paths['train'] += path_names[:int(len(path_names)*0.85)]
            image_paths['validation'] += path_names[int(len(path_names)*0.85):]
        with open(os.path.join(image_directory,'images_atlas.pkl'), 'wb') as file:
            pickle.dump(image_paths, file, pickle.HIGHEST_PROTOCOL)
        
    if not os.path.isfile(os.path.join(image_directory,'name_to_label.pkl')):

        name_to_label = {'great white shark':0, 'flamingo':1, 'banana':2, 'strawberry':3, 'pelican':4, 'tiger':5}
        with open(os.path.join(image_directory,'name_to_label.pkl'), 'wb') as file:
            pickle.dump(name_to_label, file, pickle.HIGHEST_PROTOCOL)

class FMAsmallDataset(torch.utils.data.Dataset):
    def __init__(self, directory, input_size, train=True, transform=None, force=False):
        create_FMA_small_dataset_atlas(directory)
        with open(os.path.join(directory,'audio_atlas.pkl'), 'rb') as file:
            self.audio_paths = pickle.load(file)['train'] if train else pickle.load(file)['validation']
        with open(os.path.join(directory,'name_to_label.pkl'), 'rb') as file:
            name_to_label = pickle.load(file)
        
        self.labels = [name_to_label[v.split(os.sep)[-2]] for v in self.audio_paths]
        
        self.labels_name = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        self.input_size = input_size
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        audio_path = self.audio_paths[idx]
        
        
        audio_object = pydub.AudioSegment.from_mp3(audio_path)
        audio = np.array(audio_object.get_array_of_samples())
        audio = audio.astype(np.float32)
        
        audio = audio_utils.zeropad(audio, int(2644992/3)) # 10 seconds

        label = self.labels[idx]
        
        return audio, label

class SpeechCommandDataset(torch.utils.data.Dataset):

    def __init__(self, directory, input_size, train=True, transform=None, force=False, evaluation=False):

        create_speech_commands_dataset_atlas(directory)
        with open(os.path.join(directory,'audio_atlas.pkl'), 'rb') as file:
            if train:
                self.audio_paths = pickle.load(file)['train']
            elif evaluation:
                self.audio_paths = pickle.load(file)['evaluation']
            else:
                self.audio_paths = pickle.load(file)['validation']
                

        with open(os.path.join(directory,'name_to_label.pkl'), 'rb') as file:
            name_to_label = pickle.load(file)

        self.labels = [name_to_label[v.split(os.sep)[-2]] for v in self.audio_paths]
        
        self.labels_name = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.input_size = input_size
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        fs, audio = wavread(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_utils.zeropad(audio, 16128) # 1 seconds
        
        label = self.labels[idx]
        
        return audio, label


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory, input_size, train=True, transform=None, force=False):
        create_mnist_dataset_atlas(image_directory)
        with open(os.path.join(image_directory,'mnist_atlas.pkl'), 'rb') as file:
            self.image_paths = pickle.load(file)['train'] if train else pickle.load(file)['validation']
        with open(os.path.join(image_directory,'name_to_label.pkl'), 'rb') as file:
            name_to_label = pickle.load(file)

        self.labels = [name_to_label[v.split(os.sep)[-2]] for v in self.image_paths]
        
        self.labels_name = ['0','1','2','3','4','5','6','7','8','9']
        self.input_size = input_size

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = skio.imread(image_path)
        image = sktransform.resize(image, (self.input_size, self.input_size))
        
        image = np.expand_dims(image, axis=0)
        label = self.labels[idx]
        return image, label

class DogsCatsDataset(torch.utils.data.Dataset):

    def __init__(self, image_directory, input_size, train=True, transform=None, force=False):

        create_dogs_cats_dataset_atlas(image_directory)
        with open(os.path.join(image_directory,'images_atlas.pkl'), 'rb') as file:
            self.image_paths = pickle.load(file)['train'] if train else pickle.load(file)['validation']
        self.labels = [0 if v.split(os.sep)[-1].split('.')[0] == 'cat' else 1 for v in self.image_paths]
        self.labels_name = ['cat', 'dog']
        self.input_size = input_size

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        image = skio.imread(image_path) 
        image = sktransform.resize(image, (self.input_size, self.input_size))
        image = np.swapaxes(np.swapaxes(image,0,2),1,2)
        label = self.labels[idx]
        return image, label

class SelectedImagenetDataset(torch.utils.data.Dataset):

    def __init__(self, image_directory, input_size, train=True, transform=None, force=False):

        create_imagenet_dataset_atlas(image_directory)
        with open(os.path.join(image_directory,'images_atlas.pkl'), 'rb') as file:
            self.image_paths = pickle.load(file)['train'] if train else pickle.load(file)['validation']
        with open(os.path.join(image_directory,'name_to_label.pkl'), 'rb') as file:
            name_to_label = pickle.load(file)

        self.labels = [name_to_label[v.split(os.sep)[-2]] for v in self.image_paths]
        
        self.labels_name = ['great white shark', 'flamingo', 'banana', 'strawberry', 'pelican', 'tiger']
        self.input_size = input_size
        
        

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        image = skio.imread(image_path) 
        image = sktransform.resize(image, (self.input_size, self.input_size))
        if len(image.shape) != 3 or image.shape[2] == 1:
            image = image.reshape(self.input_size, self.input_size)
            image_new = np.empty((self.input_size, self.input_size, 3))
            image_new[:,:,0] = image / 0.3
            image_new[:,:,1] = image / 0.59
            image_new[:,:,2] = image / 0.11
            image = image_new
        
        image = np.swapaxes(np.swapaxes(image,0,2),1,2)
        label = self.labels[idx]
        
        return image, label

def get_dataset(dataset_name, dataset_path, input_size = 28, train = True, evaluation=False):

    if dataset_name == 'speech':
        return SpeechCommandDataset(dataset_path, input_size, train, evaluation=evaluation)
    elif 'speech_eval' in dataset_name:
        return SpeechCommandDataset(dataset_path, input_size, train)
    elif dataset_name == 'dogscats':
        return DogsCatsDataset(dataset_path, input_size, train)
    elif dataset_name == 'imagenet':
        return SelectedImagenetDataset(dataset_path, input_size, train)
    elif dataset_name == 'mnist':
        return MnistDataset(dataset_path, input_size, train)
    elif dataset_name == 'FMA_small':
        return FMAsmallDataset(dataset_path, input_size, train)

def convert_to_rgb(im):
    dims = len(im.shape)
    im = np.squeeze(im)
    if dims == 4:
        im_new = np.zeros((im.shape[0],)+(3,)+(im.shape[1:]))
        im_new[:,0] = im / 0.3
        im_new[:,1] = im / 0.59
        im_new[:,2] = im / 0.11
    elif dims == 3:
        im_new = np.zeros((3,)+(im.shape[0:]))
        im_new[0] = im / 0.3
        im_new[1] = im / 0.59
        im_new[2] = im / 0.11
    else:
        print("Not supported size for RGB conversion")
    return im_new

#####################
#   Parsing utils   #
#####################

def get_args_train():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Adversarial attacks (training and evaluation) in Pytorch', 
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )

    # Standard parsing
    parser.add_argument('--images_dir', default = '..'+os.sep+'Figures'+os.sep,
        help = 'folder to store the resulting images')
    parser.add_argument('--models_dir', default = '..'+os.sep+'Models'+os.sep,
        help = 'folder to store the models') 
    parser.add_argument('--datasets_dir', default = '..'+os.sep+'Datasets'+os.sep,
        help = 'folder where the datasets are stored') 
    parser.add_argument('--dataset_name', choices = ['dogscats', 'imagenet','speech','mnist','FMA_small'], default = 'imagenet', 
        help = 'dataset where to run the experiments') 
    parser.add_argument('--model_name', choices = [
            'images_shufflenetv2', 'images_mobilenetv2', 'images_resnet18', 'audio_conv_raw', 'simple_dense', 'audio_M3','audio_M5', 'audio_MJ','audio_F7','audio_F7_base','audio_F10', 'audio_conv2d_mfcc',  'audio_conv2d_spectrogram'
        ], default= 'images_shufflenetv2',
        help = 'model used in the experiments')
    parser.add_argument('--n_iterations', type = int, default = 2000, 
        help = 'number of training iterations')
    parser.add_argument('--batch_size', type = int, default = 32, 
        help = 'number of samples in each batch')
    parser.add_argument('--learning_rate', type = float, default = 0.03, 
        help = 'learning rate of the training algorithm (Adam)')
    parser.add_argument('--verbose_rate', type = int, default = 250, 
        help = 'number of iterations to show preliminary results and store the model')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
        help = 'If flag is present the program will use available CUDA device')


    # Adversarial training parsing 
    parser.add_argument('--adversarial_training_algorithm', choices = [
            'none', 'FGSM_vanilla', 'PGD', 'fast', 'free','ONE_PIXEL','DE','DE_masking', 'LGAP','RGAP'
        ], default = 'none',
        help = 'adversarial training algorithm for the experiments')
    parser.add_argument('--epsilon', type = float, default = 0.03, 
        help = 'strength of the linear perturbation of the adversarial')
    parser.add_argument('--min_value_input', type = float, default = 0.0, 
        help = 'minimum value of the input variable')
    parser.add_argument('--max_value_input', type = float, default = 1.0, 
        help = 'maximum value of the input variable')
    parser.add_argument('--n_steps_adv', type = int, default = 7,
        help = 'number of steps for the iterative adversarial algorithms')

    

    return parser.parse_args()

def get_args_evaluate():
    '''
    This function returns the arguments from terminal and set them to display
    ''' 

    parser = argparse.ArgumentParser(
        description = 'Adversarial attacks (training and evaluation) in Pytorch', 
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )

    # Standard parsing
    parser.add_argument('--images_dir', default = '..'+os.sep+'Figures'+os.sep,
        help = 'folder to store the resulting images')
    parser.add_argument('--models_dir', default = '..'+os.sep+'Models'+os.sep,
        help = 'folder to store the models') 
    parser.add_argument('--datasets_dir', default = '..'+os.sep+'Datasets'+os.sep,
        help = 'folder where the datasets are stored') 
    parser.add_argument('--dataset_name', choices = ['dogscats', 'imagenet','speech','speech_eval_RG_targeted_32','speech_eval_RG_untargeted_32','speech_eval_RG_targeted_64','speech_eval_RG_untargeted_64','speech_eval_LG_targeted_32','speech_eval_LG_untargeted_32','speech_eval_LG_targeted_64','speech_eval_LG_untargeted_64','speech_eval_RG_targeted_clean','speech_eval_RG_untargeted_clean','speech_eval_LG_targeted_clean','speech_eval_LG_untargeted_clean','mnist','FMA_small'], default = 'imagenet', 
        help = 'dataset where to run the experiments') 
    parser.add_argument('--model_name', choices = [
            'images_shufflenetv2', 'images_mobilenetv2', 'images_resnet18', 'audio_conv_raw', 'simple_dense', 'audio_M3','audio_M5','audio_MJ','audio_F7','audio_F7_base','audio_F10', 'audio_conv2d_mfcc', 'audio_conv2d_spectrogram'
        ], default= 'images_shufflenetv2',
        help = 'model used in the experiments')
    parser.add_argument('--batch_size', type = int, default = 32, 
        help = 'number of samples in each batch')
    parser.add_argument('--n_samples_adv', type = int, default = 100,
        help = 'number of adversaries to generate')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
        help = 'If flag is present the program will use available CUDA device')
    parser.add_argument('--targeted', dest='targeted', action='store_true',
        help = 'Targeted attack or not')
    parser.set_defaults(feature=False)
    
    # Transferability parsing
    parser.add_argument('--adversary_model_name', choices = [
            'images_shufflenetv2', 'images_mobilenetv2', 'images_resnet18', 'audio_conv_raw', 'simple_dense', 'audio_M3','audio_M5','audio_MJ','audio_F7','audio_F7_base','audio_F10', 'audio_conv2d_mfcc', 'audio_conv2d_spectrogram'
        ], default = 'none',
        help = 'model used for generating adversarial examples when testing transferability')


    # Adversarial training parsing 
    parser.add_argument('--adversarial_training_algorithm', choices = [
            'none', 'FGSM_vanilla', 'PGD', 'fast', 'free','ONE_PIXEL','DE','DE_masking','LGAP','RGAP'
        ], default = 'none',
        help = 'adversarial training algorithm for the experiments')
    # Adversarial attack parsing 
    parser.add_argument('--adversarial_attack_algorithm', choices = [
            'none', 'FGSM_vanilla', 'PGD', 'fast', 'free','ONE_PIXEL','DE','DE_masking','LGAP','RGAP'
        ], default = 'FGSM_vanilla',
        help = 'adversarial attack algorithm for the experiments')
    parser.add_argument('--epsilon', type = float, default = 0.03, 
        help = 'strength of the linear perturbation of the adversarial')
    parser.add_argument('--min_value_input', type = float, default = 0.0, 
        help = 'minimum value of the input variable')
    parser.add_argument('--max_value_input', type = float, default = 1.0, 
        help = 'maximum value of the input variable')
    parser.add_argument('--target', type=int, default=3,
        help = 'id of the target for the targeted attack')
    parser.add_argument('--adv_verbose', dest='adv_verbose', action='store_true',
        help = 'If flag is present the program will enable verbose adversarial generation')
    parser.add_argument('--adv_parameters', dest='adv_parameters', nargs='+', default=None,
        help = 'List of parameters for AdversarialGenerator, see each algorithm for list of parameters')
    
    return parser.parse_args()


def simple_hash(words):
    letter_to_int = {'a':5, 'd':9 , 'r': 27, '_':2}
    y = 0
    for i, word in enumerate(words):
        for letter in word:
            if letter in letter_to_int:
                y += letter_to_int[letter]
        y += (len(word)*(i + 7)) * 137
    y = y % 9999
    return str(y)

