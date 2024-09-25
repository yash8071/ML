import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

def loss_fn_kd(outputs, labels, teacher_outputs, alpha , T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    log_softmax_outputs = F.log_softmax(outputs / T, dim=1)
    softmax_teacher_outputs = F.softmax(teacher_outputs / T, dim=1)
    kld_loss = nn.KLDivLoss()(log_softmax_outputs, softmax_teacher_outputs) * (alpha * T * T)
    ce_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    KD_loss = kld_loss + ce_loss
    return KD_loss

def train_lora(model, train_loader, args, val_loader, criterion, optimizer):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(args.epochs):
        train_loss, total_elements = 0, 0
        model.train()
        for inputs, masks, labels in train_loader:
            inputs, masks, labels = inputs.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            total_elements += inputs.size(0)
        train_losses.append(train_loss / total_elements)

        if epoch % 1 == 0:
            val_loss, correct_predictions, total_elements = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for inputs, masks, labels in val_loader:
                    inputs, masks, labels = inputs.to(args.device), masks.to(args.device), labels.to(args.device)
                    outputs = model(inputs, masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() 
                    correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total_elements += inputs.size(0)
            val_losses.append(val_loss / total_elements)
            accuracy = correct_predictions / total_elements
            accuracies.append(accuracy)
            print(f"At Epoch {epoch}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Accuracy: {accuracy}")
    plot_and_save_graph(range(len(train_losses)), train_losses, 'Training Loss', 'LoRA_training_loss_plot.png')
    plot_and_save_graph(range(len(val_losses)), val_losses, 'Validation Loss', 'LoRA_validation_loss_plot.png')
    plot_and_save_graph(range(len(accuracies)), accuracies, 'Accuracy', 'LoRA_accuracy_plot.png')
    return train_losses, val_losses, accuracies

def train_distil(model, teacher_model, train_loader, args, val_loader, criterion, optimizer):
    tlosses = []
    vlosses = []
    accuracy = []

    for epoch in range(args.epochs):
        total_elements = 0
        total_loss = 0
        model.train()
        for inputs, masks, labels in train_loader:
            inputs, masks, labels = inputs.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            with torch.no_grad():
                teacher_output = teacher_model(inputs, masks)
                loss += loss_fn_kd(output, labels, teacher_output, 0.5, 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 
            total_elements += inputs.size(0)
        tlosses.append(total_loss / total_elements)

        if epoch % 1 == 0:
            val_loss, correct_predictions, total_elements = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for inputs, masks, labels in val_loader:
                    inputs, masks, labels = inputs.to(args.device), masks.to(args.device), labels.to(args.device)
                    output = model(inputs)
                    loss = criterion(output, labels)
                    correct_predictions += (torch.argmax(output, dim=1) == labels).sum().item()
                    val_loss += loss.item() 
                    total_elements += inputs.size(0)
            vlosses.append(val_loss / total_elements)
            accuracy.append(correct_predictions / total_elements)
            print(f"At Epoch {epoch}, Training Loss: {tlosses[-1]}, Validation Loss: {vlosses[-1]}, Accuracy: {accuracy[-1]}")
   
    plot_and_save_graph(range(len(tlosses)), tlosses, 'Training Loss', 'distil_training_loss_plot.png')
    plot_and_save_graph(range(len(vlosses)), vlosses, 'Validation Loss', 'distil_validation_loss_plot.png')
    plot_and_save_graph(range(len(accuracy)), accuracy, 'Accuracy', 'distil_accuracy_plot.png')
    return tlosses, vlosses, accuracy

def train_rnn(model, train_loader, args, val_loader, criterion, optimizer):
    tlosses = []
    vlosses = []
    accuracy = []

    for epoch in range(args.epochs):
        total_elements = 0
        total_loss = 0
        model.train()
        for inputs, _, labels in train_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 
            total_elements += inputs.size(0)
        tlosses.append(total_loss / total_elements)

        if epoch % 1 == 0:
            val_loss, correct_predictions, total_elements = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for inputs, _, labels in val_loader:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    output = model(inputs)
                    loss = criterion(output, labels)
                    correct_predictions += (torch.argmax(output, dim=1) == labels).sum().item()
                    val_loss += loss.item() 
                    total_elements += inputs.size(0)
            vlosses.append(val_loss / total_elements)
            accuracy.append(correct_predictions / total_elements)
            print(f"At Epoch {epoch}, Training Loss: {tlosses[-1]}, Validation Loss: {vlosses[-1]}, Accuracy: {accuracy[-1]}")
    
    plot_and_save_graph(range(len(tlosses)), tlosses, 'Training Loss', 'rnn_training_loss_plot.png')
    plot_and_save_graph(range(len(vlosses)), vlosses, 'Validation Loss', 'rnn_validation_loss_plot.png')
    plot_and_save_graph(range(len(accuracy)), accuracy, 'Accuracy', 'rnn_accuracy_plot.png')
    return tlosses, vlosses, accuracy

def train(model, train_loader, val_loader, args, teacher_model=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.mode == 'LoRA':
        train_lora(model, train_loader, args, val_loader, criterion, optimizer)
    elif args.mode == 'distil':
        train_distil(model, teacher_model, train_loader, args, val_loader, criterion, optimizer)
    elif args.mode == 'rnn':
        train_rnn(model, train_loader, args, val_loader, criterion, optimizer)

def plot_and_save_graph(x, y, ylabel, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs. Epochs')
    plt.savefig(os.path.join('plots', filename))
    plt.close()