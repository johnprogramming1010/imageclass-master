import argparse
from model_cnn import Net
from model_linear import LinearNet
import torch
import os

def app_model(filename, output_file, model_type, num_classes):
    '''
    This function loads a saved model and creates a torchscript model for Android deployment.
    args: 
    filename (str) - name of the model file to load
    output_file (str) - name of the output file to save the torchscript model
    model_type (str) - type of model to use (cnn or linear)
    num_classes (int) - number of classes in the dataset    
    '''
    # get the current working directory
    cwd = os.getcwd()

    # print output to the console
    print(f">> current working directory: {cwd}")

    # make model place holder
    if model_type == 'cnn':
        model_name = Net(num_classes)
    if model_type == 'linear':
        model_name = LinearNet(num_classes)
    # Where to pull saved model from
    saved_model = (f'./completed_models/{filename}')

    # Load saved model into variable 'model_name' - print status
    model_name.load_state_dict(torch.load(saved_model))
    print(f">> model loaded from {saved_model}")

    # Place model into eval mode - print status
    model_name.eval()
    print(">> model in evaluation mode")

    # Create torchscript so it can be read on mobile
    torchscript_model = torch.jit.script(model_name)

    # Path to save torchscript model
    PATH = '../app/src/main/assets/' 
    
    if output_file is not None:
        PATH += output_file
    else:
        PATH += 'model'

    # Save torchscript model - print status
    torchscript_model.save(f"{PATH}.pt")
    print(f">> model saved to {PATH}.pt\nProgram Ended")

def description():
    '''
    Description of the program, this only runs if argument -h is used in the command line.
    args: None
    returns: None
    '''
    print(f"\nThis program creates a torchscript model for Android deployment.")
    print(f"\nIt requires the filename of the model to load and the name of the output file to save the torchscript model.")
    print(f"The type of model to use (cnn or linear) and the number of classes in the dataset are optional arguments.")
    print(f"Optional arguments are: -m (default cnn) and -n (default 100)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description(), epilog="Example use of the program would use the following command: python app_model.py -f model.pth -o model -m cnn -n 100")
    # assign a value for filename
    parser.add_argument('-f', '--filename', type=str, help='Name of the model file to load')
    # assign a value for output
    parser.add_argument('-o', '--output', type=str, help='Name of the output file to save the torchscript model')
    # assign which model to use
    parser.add_argument('-m', '--model_type', type=str, choices=['cnn', 'linear'], default='cnn', help='Type of model to use (cnn or linear)')
    # assign a value for num_classes
    parser.add_argument('-n', '--num_classes', type=int, default=100, help='Number of classes in the dataset')

    args = parser.parse_args()

    if args.filename is None:
        print("Please provide the filename of the model to load.")
    else:
        app_model(args.filename, args.output, args.model_type, args.num_classes)

