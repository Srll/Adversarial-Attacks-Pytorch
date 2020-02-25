import torch, torchvision
from skimage import io as skio
import utils 
import matplotlib.pyplot as plt 
import networks
import progressbar
import adversaries
import os

torch.manual_seed(1)

criterion = None
adversary = None
figures_path = None
# obtain the arguments 
args = utils.get_args_evaluate()

def evaluate_model(model, adversary, dataloader, labels_name, targeted=False, target_id=0, input_type = 'images'):

    accuracy = 0 
    accuracy_adversarial = 0
    loss = 0
    loss_adversarial = 0

    for _, (inputs,labels) in enumerate(progressbar.progressbar(dataloader)):
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        if targeted:
            target = torch.LongTensor((torch.ones_like(labels) * target_id))
            inputs_adversarial, adversarial_noise, labels_estimations_adversarial, labels_estimations = \
                adversary.generate_adversarial(args.adversarial_attack_algorithm, inputs, target, targeted = targeted, eps=args.epsilon)
        else:
            inputs_adversarial, adversarial_noise, labels_estimations_adversarial, labels_estimations = \
                adversary.generate_adversarial(args.adversarial_attack_algorithm, inputs, labels, targeted = targeted, eps=args.epsilon)
        loss += criterion(labels_estimations, labels)
        loss_adversarial += criterion(labels_estimations_adversarial, labels)
        accuracy += torch.mean(labels.eq(torch.max(labels_estimations,dim=1)[1]).float())
        accuracy_adversarial += torch.mean(labels.eq(torch.max(labels_estimations_adversarial ,dim=1)[1]).float())
    
    target_name = None
    if targeted:
        target_name = labels_name[target_id]
    
    # TODO fix
    input_type = 'images'
    if input_type == 'images':
        save_images(inputs, adversarial_noise, inputs_adversarial, labels, labels_estimations, labels_estimations_adversarial, 
            path=figures_path, target_name=target_name)

    loss /= len(dataloader)
    loss_adversarial /= len(dataloader)
    accuracy /= len(dataloader)
    accuracy_adversarial /= len(dataloader)

    if targeted:
        print('Targeted attack: {target_name}')
    else: 
        print('Untargeted attack:')

    print(f'Cross-Entropy Loss: {loss:.2f}')
    print(f'Cross-Entropy Loss (adversarial): {loss_adversarial:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Accuracy (adversarial): {accuracy_adversarial:.2f}')

    
def save_images(x, adv_noise, x_adv, y, y_est, y_est_adv, path, target_name=None):
    """
    def paint_images(im, correct):
        im = torch.nn.functional.pad(im,pad=(10,10,10,10),mode='constant',value=0)
        im[correct,1,0:10] = 1.0
        im[correct,1,-10:-1] = 1.0
        im[correct,1,:,0:10] = 1.0
        im[correct,1,:,-10:-1] = 1.0
        im[~correct,0,0:10] = 1.0
        im[~correct,0,-10:-1] = 1.0
        im[~correct,0,:,0:10] = 1.0
        im[~correct,0,:,-10:-1] = 1.0
        return im
    """
    
    
    def paint_images(im, correct):
        im = torch.nn.functional.pad(im,pad=(10,10,10,10),mode='constant',value=0)
        im[correct,0,0:10] = 0.0
        im[correct,0,-10:-1] = 0.0
        im[correct,0,:,0:10] = 0.0
        im[correct,0,:,-10:-1] = 0.0
        im[~correct,0,0:10] = 1.0
        im[~correct,0,-10:-1] = 1.0
        im[~correct,0,:,0:10] = 1.0
        im[~correct,0,:,-10:-1] = 1.0
        return im
    
    
    correct = y.eq(torch.max(y_est,dim=1)[1])
    correct_adv = y.eq(torch.max(y_est_adv,dim=1)[1])
    x = paint_images(x, correct)
    x_adv = paint_images(x_adv,correct_adv)
    
    tail = '' if target_name is None else '_' + target_name
    body = args.model_name + '_' + args.adversarial_training_algorithm + '_' + args.adversarial_attack_algorithm + '_'
    torchvision.utils.save_image(x,os.path.join(path,body + 'image' + tail + '.png'),nrow=8)
    torchvision.utils.save_image(adv_noise,os.path.join(path,body + 'noise' + tail + '.png'),nrow=8)
    torchvision.utils.save_image(x_adv,os.path.join(path,body + 'image_adv' + tail + '.png'),nrow=8)

    im1 = skio.imread(os.path.join(path,body + 'image' + tail + '.png'))
    im2 = skio.imread(os.path.join(path,body + 'noise' + tail + '.png'))
    im3 = skio.imread(os.path.join(path,body + 'image_adv' + tail + '.png'))
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(120,20))
    ax[0].imshow(im1)
    ax[1].imshow(im2)
    ax[2].imshow(im3)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[1].set_ylabel('+',fontsize=250,ha='right',va='center',rotation='horizontal')
    ax[2].set_ylabel('=',fontsize=250,ha='right',va='center',rotation='horizontal')
    plt.tight_layout()
    plt.savefig(os.path.join(path,body + 'resulting_image' + tail + '.png'),format='png')



def evaluate():
    # obtain the model and the datasets
    dataset_path = args.datasets_dir + args.dataset_name 
    models_path = args.models_dir + args.dataset_name 
    global figures_path
    figures_path = args.images_dir + args.dataset_name
    model = networks.CNN(args.model_name,dataset_name=args.dataset_name)
    dataset_train = utils.get_dataset(args.dataset_name, dataset_path)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataset_eval = utils.get_dataset(args.dataset_name, dataset_path, train=False)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    labels_name = dataset_eval.labels_name

    # obtain the criterion used for training 
    global criterion
    criterion = torch.nn.CrossEntropyLoss()

    # load the checkpoint, if any 
    checkpoint_path = os.path.join(models_path, args.model_name + '_' + args.adversarial_training_algorithm + '.chkpt')
    if os.path.isfile(checkpoint_path):  
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        iteration = checkpoint['iteration'] + 1
        print(f'Model trained with {iteration} iterations')
        print(f'Loss at iteration {iteration} -> train: {checkpoint["training_loss"]:.3f} | eval: {checkpoint["evaluation_loss"]:.3f}')
        print(f'Accuracy at iteration {iteration} -> train: {checkpoint["training_accuracy"]:.3f} | eval: {checkpoint["evaluation_accuracy"]:3f}')

    # generate the adversary
    #global adversary
    adversary = adversaries.AdversarialGenerator(model,criterion)


    # evaluate the model 
    input_type = args.model_name.split('_')[0]
    evaluate_model(model, adversary, dataloader_eval, labels_name, targeted=False, input_type=input_type)
    #evaluate_model(model, adversary, dataloader_eval, labels_name, targeted=True, target_id=args.target, input_type=input_type)

if __name__ == "__main__":
    evaluate()