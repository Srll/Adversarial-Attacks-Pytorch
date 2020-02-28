import torch 
import utils 
import matplotlib.pyplot as plt 
import networks
import progressbar
import adversaries
import os
import preprocess



torch.manual_seed(1)

def train():
    
    # obtain the arguments 
    args = utils.get_args_train()

    # obtain the model and the datasets
    dataset_path = args.datasets_dir + args.dataset_name 
    models_path = args.models_dir + args.dataset_name 
    #if args.model_name == "simple_dense":
    #    model = networks.Simple_dense(10)
    #else:
    
    model = networks.CNN(args.model_name,dataset_name=args.dataset_name,preprocess_sequence=args.preprocess_sequence)
    dataset_train = utils.get_dataset(args.dataset_name, dataset_path)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataset_eval = utils.get_dataset(args.dataset_name, dataset_path, train=False)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    # prepare the optimization  
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
            params_to_update.append(param)
    learning_rate = args.learning_rate
    
    if args.adversarial_training_algorithm == 'FGSM_vanilla':
        optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)
    else: 
        #optimizer = torch.optim.SGD(params_to_update, lr=0.01,momentum=0.5)
        optimizer = torch.optim.Adam(params_to_update, lr=0.01)

    # prepare the losses and iterations
    n_iterations = args.n_iterations 
    n_iterations_show = args.verbose_rate
    running_loss = 0
    running_accuracy = 0
    iteration = 0
    bar = progressbar.ProgressBar(max_value=100)

    # load the checkpoint, if any 
    checkpoint_path = os.path.join(models_path, args.model_name + '_' + args.adversarial_training_algorithm + '_' + utils.simple_hash(args.preprocess_sequence) + '.chkpt')
    if os.path.isfile(checkpoint_path):  
        checkpoint = torch.load(checkpoint_path)
        model.preprocess = checkpoint['preprocessing_sequence']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration'] + 1
        print(f'Starting the model at iteration {iteration + 1}')
        print(f'Loss at iteration {iteration} -> train: {checkpoint["training_loss"]:.3f} | eval: {checkpoint["evaluation_loss"]:.3f}')
        print(f'Accuracy at iteration {iteration} -> train: {checkpoint["training_accuracy"]:.3f} | eval: {checkpoint["evaluation_accuracy"]:3f}')
    else: 
        print(f'Starting the model at iteration {iteration + 1}')

    # prepare adversary
    adversary = adversaries.AdversarialGenerator(model,criterion)
    
    bar = progressbar.ProgressBar(max_value=n_iterations_show)
    # train the model
    while iteration < n_iterations:
        for _, (inputs,labels) in enumerate(dataloader_train):
            
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            inputs_adv = adversary.generate_adversarial(args.adversarial_training_algorithm, inputs, labels, 
                    eps=args.epsilon, x_min=args.min_value_input, x_max=args.max_value_input, alpha=learning_rate, train=True)
            
            optimizer.zero_grad()
            
            model.zero_grad()
            labels_estimations = model(inputs_adv)
            loss = criterion(labels_estimations, labels)
            loss.backward()
            #print(loss.grad)
            #print(model)
            #print(model.fc1.weight.grad) 
            optimizer.step()
            

            # Show statistics and percentage
            with torch.no_grad():
                running_loss += loss.item()
                running_accuracy += torch.mean(labels.eq(torch.max(labels_estimations,dim=1)[1]).float())
                bar.update(iteration % n_iterations_show)
                #bar.update(int(100 * i / n_iterations_show))
                if iteration % n_iterations_show == n_iterations_show-1:
                    
                    loss_eval = 0.0
                    accuracy_eval = 0.0
                    print('\nEvaluating...')
                    for _, (inputs,labels) in enumerate(progressbar.progressbar(dataloader_eval)):
                            inputs = inputs.type(torch.FloatTensor)
                            labels = labels.type(torch.LongTensor)
                            labels_estimations = model(inputs)
                            loss_eval += criterion(labels_estimations, labels)
                            accuracy_eval += torch.mean(labels.eq(torch.max(labels_estimations,dim=1)[1]).float())
                    
                    print(f'Loss at iteration {iteration+1} -> train: {running_loss/n_iterations_show:.3f} | eval: {loss_eval/len(dataloader_eval):.3f}')
                    print(f'Accuracy at iteration {iteration+1} -> train: {running_accuracy/n_iterations_show:.3f} | eval: {accuracy_eval/len(dataloader_eval):.3f}')

                    torch.save({
                        'preprocessing_sequence': model.preprocess,
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_loss': running_loss/n_iterations_show,
                        'evaluation_loss': loss_eval/len(dataloader_eval), 
                        'training_accuracy': running_accuracy/n_iterations_show,
                        'evaluation_accuracy': accuracy_eval/len(dataloader_eval)
                        }, checkpoint_path)

                    
                    running_loss = 0.0
                    running_accuracy = 0.0

            iteration += 1


# To make multiprocessing work on windows
if __name__ == '__main__':
    train()