import torch
from argparse import Namespace

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import *
from LogisticRegression.model import SoftmaxRegression as LinearClassifier
from ContrastiveRepresentation.model import Encoder, Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model


def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''
    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    classes = len(np.unique(y_train))
    
    # Create the model
    encoder = Encoder(args.z_dim).to(ptu.device)
    if args.mode == 'fine_tune_linear':
        classifier = LinearClassifier(args.z_dim, classes)
    elif args.mode == 'fine_tune_nn':
        classifier = Classifier(args.z_dim, classes)
        classifier = classifier.to(ptu.device)
    
    if args.mode == 'cont_rep':
        # Fit the contrastive model
        X_val_tensor = torch.tensor(X_val).float()
    
        # Load encoder state dictionary
        encoder.load_state_dict(torch.load(args.encoder_path))
        
        # Evaluate Encoder on validation set
        encoder.eval()
        z = None
        batch_size = args.batch_size
        for i in range(0, len(X_val_tensor), batch_size):
            if i + batch_size > len(X_val_tensor):
                X_batch = X_val_tensor[i:].to(encoder.device)
            else:
                X_batch = X_val_tensor[i:i+batch_size].to(encoder.device)
            repr_batch = encoder(X_batch)
            repr_batch = repr_batch.cpu().detach().numpy()
            if z is None:
                z = repr_batch
            else:
                z = np.concatenate((z, repr_batch))
            
        # Plot the t-SNE after fitting the encoder
        plot_tsne(z, y_val)
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        encoder.load_state_dict(torch.load(args.encoder_path))
        X_train = ptu.from_numpy(X_train)
        y_train = ptu.from_numpy(y_train, torch.long)
        X_val = ptu.from_numpy(X_val)
        y_val = ptu.from_numpy(y_val, torch.long)
        
        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        
        # Plot the losses
        # print(type(train_losses[0]), type(test_losses[0]), type(train_accs[0]), type(test_accs[0]))
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()

        # Save the predictions for the test data in a CSV file
        encoder.eval()
        if args.mode == 'fine_tune_nn':
            classifier.eval()
        y_preds = []
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i+args.batch_size].to(ptu.device)
            repr_batch = encoder(X_batch)
            if 'linear' in args.mode:
                repr_batch = ptu.to_numpy(repr_batch)
            y_pred_batch = classifier(repr_batch)
            if 'nn' in args.mode:
                y_pred_batch = ptu.to_numpy(y_pred_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv',\
                y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv')
