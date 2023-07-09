import re
import matplotlib.pyplot as plt

'''
This script creates training and validation curves from the log output of RfDNet.
In contrast to the sub-loss-visualization this script only prints the total loss.
'''

def save_loss_plot(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss_plot.png')
    plt.show()

def extract_loss_from_file(file_path, mode='train'):
    losses = []
    pattern = fr'Currently the last {mode} loss \(total\) is: (\d+\.\d+)'

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)

    return losses

train_losses = extract_loss_from_file(r'C:\Users\Besitzer\Desktop\adl4cv\trainloss.txt', mode='train')
val_losses = extract_loss_from_file(r'C:\Users\Besitzer\Desktop\adl4cv\trainloss.txt', mode='val')

# make sure the losses have the same size
losses_size = min(len(train_losses), len(val_losses))
train_losses = train_losses[:losses_size]
val_losses = val_losses[:losses_size]

save_loss_plot(train_losses, val_losses)