import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(y_arr, title, loss_or_acc=""):
    """
    Function to plot a learning curve given train_loss and val_loss
    :param y_arr: array storing training loss or training accuracy or validation loss or validation accuracy
    :param title: string storing what type of value is stored in y_arr to be used in plot title
    :param loss_or_acc: string ylabel for plot, values: "Loss" or "Accuracy"
    """
    # Get number of epochs
    epochs = np.arange(0,len(y_arr))
    # Plot the curve and set title, legend, labels appropriately
    plt.plot(epochs, y_arr)
    plt.title(title+" vs Epoch curve")
    plt.xlabel("Epochs")
    plt.ylabel(loss_or_acc)
    plt.savefig(title.replace(" ", "_").replace(":", "")+'.png')
    plt.close()


def visualize(model_name, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Function to plot
    :param model_name: string of name of model being visualized
    :param train_loss:
    :param train_accuracy:
    :param val_loss:
    :param val_accuracy:
    :return:
    """
    plot_learning_curve(train_loss, model_name+': Training loss', 'Loss')
    plot_learning_curve(train_accuracy, model_name+': Training accuracy', 'Accuracy')
    plot_learning_curve(val_loss, model_name+': Validation loss', 'Loss')
    plot_learning_curve(val_accuracy, model_name+': Validation accuracy', 'Accuracy')