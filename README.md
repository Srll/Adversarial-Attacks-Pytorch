
# Adversarial-Attacks-Pytorch

## Table of contents
* [Project info](#general-info)
* [Setup](#setup)
* [Training Classifier](#setup)
* [Evaluate AEs](#Evaluate AEs)

## Project info
This is a master thesis project with the goal of exploring the possibility of generating inaudible adversarial perturbations in a black-box setting.
    
The MSc thesis publication of the project is available at [Fix this reference]().

A demo of some of the generated adversarial examples are available on this [GitHub page](https://srll.github.io/Adversarial-Attacks-Pytorch/).
    
	
## Setup
Start by downloading the speech-command dataset [https://arxiv.org/abs/1804.03209], e.g. from [http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

The speech-command dataset should be formatted as:

 
    Adversarial-Attacks-Pytorch
        ├── Datasets
            ├── speech
            │   ├── down
            │   ├── go
            │   ├── left
            │   ├── no
            │   ├── off
            │   ├── on
            │   ├── right
            │   ├── stop
            │   ├── up
            │   ├── yes
        ├── Models
        ├── Figures
        ├── src
        
Where each folder contain all of the available data for each label.
Note that in the thesis the 10-class speech-command dataset is used, i.e. only the 10 classes listed.

#### (The project also supports the dataset FMA: A Dataset For Music Analysis [https://arxiv.org/abs/1612.01840](https://arxiv.org/abs/1612.01840). However, there are no experimental results published for this dataset in the thesis due to the computational cost associated with training and evaluating this dataset.)

The required python packages (except torch) are listed in requirements.txt
    
    pip install -r requirements.txt
        
## Training Classifier
To train the classifiers equivalently to what was done for the experimental results in the thesis

    
Classifier name in thesis: F7,  θ0
    
    py train.py --dataset_name speech --batch_size 50 --n_iterations 20000 --gpu --verbose_rate  368 --model_name audio_F7_base

Classifier name in thesis: F7, θ1

    py train.py --dataset_name speech --batch_size 50 --n_iterations 20000 --gpu --verbose_rate  368 --model_name audio_F7

Classifier name in thesis: F10, θ2

    py train.py --dataset_name speech --batch_size 50 --n_iterations 20000 --gpu --verbose_rate  368 --model_name audio_F10

There are plenty of other settings available for training the classifiers accessible through the CLI, e.g. adversarial training, adversarial training settings, learning rate. For a full list see

    py train.py -h

## Evaluate AEs

### **Untargeted Attack**
    py evaluate.py --dataset_name speech --model_name audio_F7_base --batch_size 100 --n_samples 100 --adversarial_attack_algorithm LGAP --epsilon 128 --adv_parameters 20 1000 --targeted --gpu

### **Targeted Attack**
    py evaluate.py --dataset_name speech --model_name audio_F7_base --batch_size 100 --n_samples 100 --adversarial_attack_algorithm LGAP --epsilon 128 --adv_parameters 20 1000 --targeted --gpu


### **Transferability**
Transferability can be evaluated in two ways, either by generating new adversarial examples as done by:

    py evaluate.py --dataset_name speech --model_name audio_F7 --adversary_model_name audio_F7_base --batch_size 100 --n_samples 100 --adversarial_attack_algorithm LGAP --epsilon 128 --adv_parameters 20 1000 --targeted --gpu

In which the adversarial examples are generated using the model **--adversary_model_name**, but evaluated on model **--model_name**.


### Alternative method:
**NOTE: This will not work for experiments other than those in the thesis**

To evaluate transferability on already existing adversarial examples generated in the targeted/untargeted attacks above, one needs to save the generated adversarial examples from the targeted/untargeted attacks as new datasets. This is more complicated than the transferability evaluation above but was done to during the experimental results in the thesis.

To convert the targeted attack LGAP attack with epsilon 64, run the following commands:
    
    py tools/create_dataset.py ..\Figures\speech\audio_F7\LGAP\targeted\64.0\I

    py tools/create_dataset_clean.py ..\Figures\speech\audio_F7\LGAP\targeted\64.0\I


Once the datasets have been generated one can run the following command in order to evaluate the transferability on the architecture F10

    py evaluate.py --model_name audio_F10 --gpu --dataset_name speech_eval_LG_untargeted_64 --batch_size 10 --n_samples_adv 100 --adversarial_attack_algorithm none
    
