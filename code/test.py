'''
File used must be in .\code\completedmodels\
First Delimiter: at first underscore
    model type: cnn or linear
    _cnn or _linear
Second Delimiter: at last underscore
    number of classes: number of classes
    _nc- # classes
The rest tells you what model you are testing
EX:
python test.py -f1 model_cnn_eps-16_lr-5_nc-100 -f2 model_cnn_eps-15_lr-5_nc-20
or
python test.py -f1 model_cnn_eps-16_lr-5_nc-100
or 
python test.py -f1 model_cnn_eps-15_lr-5_nc-20
FILENAME: model_VALUE_eps-VALUE_lr-VALUE_nc-VALUE
                ^         ^        ^        ^
TEMPLATE:
python test.py model_VALUE_eps-VALUE_lr-VALUE_nc-VALUE
                     ^         ^        ^        ^
    '''

import os
import numpy as np
import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from model_cnn import Net  # Assuming your CNN model is defined in a separate file named 'model_cnn.py'
from model_linear import LinearNet  # Assuming your linear model class is defined as 'LinearNet' in 'model_linear.py'
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def convert_to_20_classes(labels):
    """
    Function to convert CIFAR100 labels to 20 classes.
    Args:
    labels: List of CIFAR100 labels.
    Returns:
    List of 20 class labels.
    """
    # Mapping from fine to coarse labels
    fine_to_coarse = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                      3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                      6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                      0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                      5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                      16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                      10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                      2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                      16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                      18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

    # Create a new list of labels
    new_labels = []
    # Convert each label to the new label
    for label in labels:
        new_labels.append(fine_to_coarse[label])

    return new_labels

def create_superclass_labels():
    '''
    Function to create a mapping between superclass labels and their corresponding names.

    Returns:
    superclass_labels: Dictionary mapping superclass labels to their names.    
    '''
    superclass_labels = {
        0: "aquatic_mammals",
        1: "fish",
        2: "flowers",
        3: "food_containers",
        4: "fruit_and_vegetables",
        5: "household_electrical_devices",
        6: "household_furniture",
        7: "insects",
        8: "large_carnivores",
        9: "large_man-made_outdoor_things",
        10: "large_natural_outdoor_scenes",
        11: "large_omnivores_and_herbivores",
        12: "medium_mammals",
        13: "non-insect_invertebrates",
        14: "people",
        15: "reptiles",
        16: "small_mammals",
        17: "trees",
        18: "vehicles_1",
        19: "vehicles_2"
    }

    return superclass_labels

def create_superclass_to_subclasses():
    '''
    Function to create a mapping between superclass labels and their corresponding fine-grained class labels.

    Returns:
    superclass_to_subclasses: Dictionary mapping superclass labels to their corresponding fine-grained class labels.    
    '''
    superclass_to_subclasses = {
        0 : ["beaver", "dolphin", "otter", "seal", "whale"],
        1 : ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        2 : ["orchid", "poppy", "rose", "sunflower", "tulip"],
        3 : ["bottle", "bowl", "can", "cup", "plate"],
        4 : ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        5 : ["clock", "keyboard", "lamp", "telephone", "television"],
        6 : ["bed", "chair", "couch", "table", "wardrobe"],
        7 : ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        8 : ["bear", "leopard", "lion", "tiger", "wolf"],
        9 : ["bridge", "castle", "house", "road", "skyscraper"],
        10 : ["cloud", "forest", "mountain", "plain", "sea"],
        11 : ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        12 : ["fox", "porcupine", "possum", "raccoon", "skunk"],
        13 : ["crab", "lobster", "snail", "spider", "worm"],
        14 : ["baby", "boy", "girl", "man", "woman"],
        15 : ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        16 : ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        17 : ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        18 : ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        19 : ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }
    return superclass_to_subclasses

def create_subclass_labels():
    '''
    Function to create a mapping between fine-grained class labels and their corresponding superclass labels.

    Returns:
    sublabels_to_subclass: Dictionary  
    '''
    sublabels_to_subclass = {
        0: 'apple',
        1: 'aquarium_fish',
        2: 'baby',
        3: 'bear',
        4: 'beaver',
        5: 'bed',
        6: 'bee',
        7: 'beetle',
        8: 'bicycle',
        9: 'bottle',
        10: 'bowl',
        11: 'boy',
        12: 'bridge',
        13: 'bus',
        14: 'butterfly',
        15: 'camel',
        16: 'can',
        17: 'castle',
        18: 'caterpillar',
        19: 'cattle',
        20: 'chair',
        21: 'chimpanzee',
        22: 'clock',
        23: 'cloud',
        24: 'cockroach',
        25: 'couch',
        26: 'crab',
        27: 'crocodile',
        28: 'cup',
        29: 'dinosaur',
        30: 'dolphin',
        31: 'elephant',
        32: 'flatfish',
        33: 'forest',
        34: 'fox',
        35: 'girl',
        36: 'hamster',
        37: 'house',
        38: 'kangaroo',
        39: 'keyboard',
        40: 'lamp',
        41: 'lawn_mower',
        42: 'leopard',
        43: 'lion',
        44: 'lizard',
        45: 'lobster',
        46: 'man',
        47: 'maple_tree',
        48: 'motorcycle',
        49: 'mountain',
        50: 'mouse',
        51: 'mushroom',
        52: 'oak_tree',
        53: 'orange',
        54: 'orchid',
        55: 'otter',
        56: 'palm_tree',
        57: 'pear',
        58: 'pickup_truck',
        59: 'pine_tree',
        60: 'plain',
        61: 'plate',
        62: 'poppy',
        63: 'porcupine',
        64: 'possum',
        65: 'rabbit',
        66: 'raccoon',
        67: 'ray',
        68: 'road',
        69: 'rocket',
        70: 'rose',
        71: 'sea',
        72: 'seal',
        73: 'shark',
        74: 'shrew',
        75: 'skunk',
        76: 'skyscraper',
        77: 'snail',
        78: 'snake',
        79: 'spider',
        80: 'squirrel',
        81: 'streetcar',
        82: 'sunflower',
        83: 'sweet_pepper',
        84: 'table',
        85: 'tank',
        86: 'telephone',
        87: 'television',
        88: 'tiger',
        89: 'tractor',
        90: 'train',
        91: 'trout',
        92: 'tulip',
        93: 'turtle',
        94: 'wardrobe',
        95: 'whale',
        96: 'willow_tree',
        97: 'wolf',
        98: 'woman',
        99: 'worm'
    }
    return sublabels_to_subclass

def calculate_confusion_matrix(all_true_labels, all_predicted_labels, num_classes):
    '''
    Function to plot the confusion matrix.
    Args:
    all_true_labels: List of true labels.
    all_predicted_labels: List of predicted labels.
    num_classes: The number of classes for the model

    returns:
    The confusion matrix for the model
    '''
    # Create the superclass to subclasses mapping
    superclass_to_subclasses = create_superclass_to_subclasses()
    sublabels_to_subclass = create_subclass_labels()

    # Create a confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Iterate over each label to calculate accuracy
    for predicted_label, true_label in zip(all_predicted_labels, all_true_labels):
        if (num_classes == 20):
            confusion_matrix[true_label][predicted_label] += 1

        else:
            true_subclass = sublabels_to_subclass[true_label]
            predicted_subclass = sublabels_to_subclass[predicted_label]

            true_superclass = None
            predicted_superclass = None
            true_subclass_number = None
            predicted_subclass_number = None

            for superclass, subclasses in superclass_to_subclasses.items():
                if true_subclass in subclasses:
                    # Extract the number from the subclass label names
                    true_subclass_number = extract_number_from_label(true_subclass, sublabels_to_subclass)
                    true_superclass = superclass
                if predicted_subclass in subclasses:
                    # Extract the number from the subclass label names
                    predicted_subclass_number = extract_number_from_label(predicted_subclass, sublabels_to_subclass)
                    predicted_superclass = superclass

            if true_superclass is not None and predicted_superclass is not None:                
                confusion_matrix[true_subclass_number][predicted_subclass_number] += 1

    return confusion_matrix

def extract_number_from_label(label, sublabels_to_subclass):
    '''
    Function to get the subclass number by subclass name.
    Args:
    label: current subclass name.
    sublabels_to_subclass: Dictionary

    returns:
    The subclass number 
    '''
    for key, value in sublabels_to_subclass.items():
        if value == label:
            return key
    return None

def compute_class_accuracies(all_predicted_labels, all_true_labels, num_classes):
    '''
    Function to compute accuracy for each subclass.
    Args:
    all_predicted_labels: List of predicted labels.
    all_true_labels: List of true labels.
    Returns:
    subclass_accuracies: List of accuracies for each subclass.
    '''

   # Create a mapping between superclass labels and their corresponding fine-grained class labels
    if (num_classes == 20):
        classLabels = create_superclass_labels()
    if (num_classes == 100):
        classLabels = create_subclass_labels()

    # Initialize a list to store class accuracies
    class_info = []

    # Iterate over each label to calculate accuracy
    for predicted_label, true_label in zip(all_predicted_labels, all_true_labels):

        class_name = classLabels[true_label]

        correct = 1 if predicted_label == true_label else 0

        class_info.append({
            'class': class_name,
            'correct': correct,
            'total': 1
        })

    # Initialize a list to store class accuracies
    class_accuracies = []

    # Iterate over each class to calculate accuracy
    while class_info:
        class_name = class_info[0]['class']
        class_instances = [item for item in class_info if item['class'] == class_name]
        total_correct = sum(item['correct'] for item in class_instances)
        total_total = sum(item['total'] for item in class_instances)
        accuracy = total_correct / total_total if total_total != 0 else -1

        class_accuracies.append({
            'class': class_name,
            'correct': total_correct,
            'total': total_total,
            'accuracy': accuracy
        })

        class_info = [item for item in class_info if item['class'] != class_name]
    
    return class_accuracies

def compute_superclass_accuracies(all_predicted_labels, all_true_labels):
    '''
    Function to compute accuracy for each superclass.
    Args:
    all_predicted_labels: List of predicted labels.
    all_true_labels: List of true labels.
    Returns:
    superclass_accuracies: List of accuracies for each superclass.    
    '''
    # Create the superclass to subclasses mapping
    superclass_to_subclasses = create_superclass_to_subclasses()
    sublabels_to_subclass = create_subclass_labels()
    superclass_labels = create_superclass_labels()

    # Initialize a list to store superclass accuracies
    superclass_accuracies = []

    # Iterate over each superclass
    for superclass, subclasses in superclass_to_subclasses.items():
        total_correct = 0
        total_total = 0

        # Iterate over each label to calculate accuracy
        for predicted_label, true_label in zip(all_predicted_labels, all_true_labels):
            if sublabels_to_subclass[true_label] in subclasses:  # Check if the true label belongs to the current superclass
                total_total += 1  # Increment total count
                if predicted_label == true_label:  # Check if prediction is correct
                    total_correct += 1  # Increment correct count

        # Calculate the accuracy for the superclass
        if total_total != 0:
            accuracy = total_correct / total_total
        else:
            accuracy = 0  # Set accuracy to 0 if no predictions were made for the superclass

        # Store accuracy in the list
        superclass_accuracies.append({
            'superclass': superclass_labels[superclass],
            'correct': total_correct,
            'total': total_total,
            'accuracy': accuracy
        })

    return superclass_accuracies

def compute_class_metrics(all_true_labels, all_predicted_labels, num_classes):
    '''
    Function to compute precision, recall, and f1 score for each superclass and macro-averaged scores for 20 classes.
    Args:
    all_true_labels: List of true labels.
    all_predicted_labels: List of predicted labels.
    Returns:
    class_metrics_list_sub: List of metrics for each subclass
    macro_precision_sub: Macro-averaged precision for subclass 
    macro_recall_sub: Macro-averaged recall for subclass
    macro_f1_sub: Macro-averaged f1 score for subclass
    class_metrics_list_super: List of metrics for each superclass
    macro_precision_super: Macro-averaged precision for superclass
    macro_recall_super: Macro-averaged recall for superclass
    macro_f1_super: Macro-averaged f1 score for superclass
    superclass_accuracies: superclass accueacies
    '''
    labels_super = create_superclass_labels()
    labels_sub = create_subclass_labels()
    superclass_accuracies = None
    class_metrics_list_super = macro_precision_super = macro_recall_sub = macro_f1_super = class_metrics_list_sub = macro_precision_sub = macro_recall_sub = macro_f1_sub = None
    all_predicted_labels_super = convert_to_20_classes(all_predicted_labels)
    all_true_labels_super = convert_to_20_classes(all_true_labels)
    if (num_classes == 20):
        class_metrics_list_super, macro_precision_super, macro_recall_super, macro_f1_super = compute_metrics(labels_super, all_true_labels, all_predicted_labels)
    if (num_classes == 100):
        class_metrics_list_super, macro_precision_super, macro_recall_super, macro_f1_super = compute_metrics(labels_super, all_true_labels_super, all_predicted_labels_super)
        class_metrics_list_sub, macro_precision_sub, macro_recall_sub, macro_f1_sub = compute_metrics(labels_sub, all_true_labels, all_predicted_labels)
    superclass_accuracies = compute_superclass_accuracies(all_predicted_labels, all_true_labels)

    return class_metrics_list_sub, macro_precision_sub, macro_recall_sub, macro_f1_sub, class_metrics_list_super, macro_precision_super, macro_recall_super, macro_f1_super, superclass_accuracies

def compute_metrics(labels, all_true_labels, all_predicted_labels):
    '''
    Function to compute precision, recall, and f1 score for each superclass and macro-averaged scores for 20 classes.
    Args:
    all_true_labels: List of true labels.
    all_predicted_labels: List of predicted labels.
    Returns:
    class_metrics_list: metrics list
    macro_precision: average precision
    macro_recall: average recall
    macro_f1: average f1
    '''
    class_metrics_list = []
    for class_name in labels.values():
        tp = fn = fp = 0
        
        for true_label, predicted_label in zip(all_true_labels, all_predicted_labels):
            if labels[true_label] == class_name and labels[predicted_label] == class_name:
                tp += 1
            if labels[true_label] == class_name and labels[predicted_label] != class_name:
                fn += 1
            if labels[true_label] != class_name and labels[predicted_label] == class_name:
                fp += 1

        # Compute precision, recall, and f1 score
        if (tp == 0):
            precision = recall = f1_score = 0
            class_metrics_list.append([class_name, tp, fn, fp, precision, recall, f1_score])
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        
        # Append subclass metrics to the list
        class_metrics_list.append([class_name, tp, fn, fp, precision, recall, f1_score])

    # Initialize lists to store precision, recall, and f1 scores of each subclass
    precisions = []
    recalls = []
    f1_scores = []

    # Iterate through subclass metrics list to collect scores
    for subclass_metrics in class_metrics_list:
        precision = subclass_metrics[4]  # Index 4 corresponds to precision in the list
        recall = subclass_metrics[5]     # Index 5 corresponds to recall in the list
        f1 = subclass_metrics[6]         # Index 6 corresponds to f1 score in the list

        # Append scores to respective lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Calculate macro-averaged scores
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return class_metrics_list, macro_precision, macro_recall, macro_f1

def test(model, testloader, device, num_classes):
    """
    Function to evaluate the model on the test dataset.

    Args:
    model: The trained neural network model.
    testloader: DataLoader for the test dataset.
    device: Device on which to perform the evaluation (e.g., 'cuda' for GPU or 'cpu' for CPU).

    Returns:
    test_accuracy: accuracy of the model
    all_predicted_labels: all labels predicted
    all_true_labels: all actual labels
    all_true_labels_super: all actual labels for 100 converted to 20 superclasses
    """
    # Set the model in evaluation mode
    model.eval()

    # Initialize lists to store predicted and true labels
    all_predicted_labels = []
    all_true_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if (num_classes == 20):
                # Convert labels to 20 classes
                labels = torch.tensor(np.array(convert_to_20_classes(labels))).long()

            # Forward pass
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append predicted and true labels to lists
            all_predicted_labels.append(predicted.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy().flatten())

        # Flatten the lists
        all_predicted_labels = np.concatenate(all_predicted_labels)
        all_true_labels = np.concatenate(all_true_labels)

        # calculate test accuracy
        test_accuracy = correct / total

        return test_accuracy, all_predicted_labels, all_true_labels

def load_model(model_filename, debug):
    """
    Function to load a trained model based on the provided filename and model type.

    Args:
    model_filename (str): The filename of the model to load.
    model_type (str): Type of the model ('cnn' or 'linear').

    Returns:
    model: The loaded model.
    model_type: cnn or linear
    num_classes: number of classes of the model
    """
    try:

        filename_parts = model_filename.split('_')

        # Extract model type from the filename
        model_type = model_filename.split('_')[1]

        num_classes = int(filename_parts[-1].split('-')[-1])  # Extract the number of classes from the filename
        
        # Define the path to the directory containing the models
        model_directory = 'completed_models'
        if debug:
            model_directory = 'code/completed_models'
            # print("all files in the directory: ", os.listdir())

        # Get a list of all model files in the directory
        model_files = os.listdir(model_directory)

        # Check if the provided filename matches any model file name (without extension)
        for file in model_files:
            if model_filename == os.path.splitext(file)[0]:
                model_path = os.path.join(model_directory, file)
                break
        else:
            raise FileNotFoundError(f"Model file '{model_filename}' not found.")

        # Initialize the model based on the model type
        if model_type == 'cnn':
            model = Net(num_classes)  # Assuming Net is the name of your CNN model class
        elif model_type == 'linear':
            model = LinearNet(num_classes)  # Assuming LinearNet is the name of your linear model class
        else:
            raise ValueError("Invalid model type. Use 'cnn' or 'linear'.")

        # Load the saved state dictionary
        state_dict = torch.load(model_path)

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

        return model, model_type, num_classes
    
    except Exception as e:
        print(f"Error occurred while loading the model: {str(e)}")
        return None, None
    

def calculate_average_accuracy(superclass_accuracies):
    """
    Function to calculate average accuracies

    Args:
    superclass_accuracies: accuraices of all the classes

    Returns:
    average_superclass_accuracy: average of accuracies
    """
    # Calculate the average superclass accuracy
    total_superclass_accuracies = len(superclass_accuracies)
    sum_superclass_accuracies = sum(item['accuracy'] for item in superclass_accuracies)
    average_superclass_accuracy = sum_superclass_accuracies / total_superclass_accuracies
    return average_superclass_accuracy

def evaluate(model, model_type, num_classes, testloader, device, criterion):
    '''
    Function to evaluate the model on the test dataset and print the results.
    Args:
    model: The trained neural network model.
    model_type: The type of the model ('cnn' or 'linear').
    num_classes: The number of classes in the dataset.
    testloader: DataLoader for the test dataset.
    device: Device on which to perform the evaluation (e.g., 'cuda' for GPU or 'cpu' for CPU).
    criterion: The loss function used for training the model.

    Returns:
    confusion matrix of the model
    average_superclass_accuracy_100: finds the average accuracy of a 100 class model
    average_superclass_accuracy_20: finds the average accuracy of a 20 class model
    '''
    # Call the test function on model
    test_accuracy, all_predicted_labels, all_true_labels = test(model, testloader, device, num_classes)

    # Call compute_class_accuracies
    class_accuracies = compute_class_accuracies(all_predicted_labels, all_true_labels, num_classes)

    # Call compute_classs_metrics function
    class_metrics_list_sub, macro_precision_sub, macro_recall_sub, macro_f1_sub, class_metrics_list_super, macro_precision_super, macro_recall_super, macro_f1_super, superclass_accuracies = compute_class_metrics(all_true_labels, all_predicted_labels, num_classes)

    print("\nModel Type: ", model_type)
    print("Number of classes", num_classes)
    # Print the test accuracy
    print(f'Test Accuracy on superclass: {test_accuracy:.2%}')

    tabledata_20 = []
    tabledata = []
    tabledata_metrics_sub = []  # Table data for subclass metrics
    tabledata_metrics_super = []  # Table data for superclass metrics
    average_superclass_accuracy_100 = average_superclass_accuracy_20 = None

    if(num_classes == 20):
        class_metrics_list = class_metrics_list_super
    if(num_classes == 100):
        class_metrics_list = class_metrics_list_super
        # print super class accuracies
        print("\nSuper class Accuracies:")
        sorted_superclass_accuracies = sorted(superclass_accuracies, key=lambda x: x['superclass'])
        for item in sorted_superclass_accuracies:
            tabledata_20.append([item['superclass'], item['accuracy']])

        max_len = max([len(row[0]) for row in tabledata_20])

        for row in tabledata_20:
            print(f"{row[0]:<{max_len}}: {row[1]:6.2%}")

        # Calculate the average superclass accuracy
        average_superclass_accuracy_100 = calculate_average_accuracy(sorted_superclass_accuracies)

        # Print the average superclass accuracy
        print(f"\nAverage Superclass Accuracy: {average_superclass_accuracy_100:.2%}")

        # Print precision, recall, and F1-score for each super-class
        print("\nPer-SuperClass Metrics:")
        sorted_class_metrics_list = sorted(class_metrics_list_super, key=lambda x: x[0])
        for item in sorted_class_metrics_list:
            tabledata_metrics_super.append([item[0], item[4], item[5], item[6]])

        max_len = max([len(row[0]) for row in tabledata_metrics_super])
        for row in tabledata_metrics_super:
            print(f"{row[0]:<{max_len}}: Precision={row[1]:6.2%}, Recall={row[2]:6.2%}, F1-score={row[3]:6.2%}")

        # Print macro-averaged precision, recall, and F1-score
        print("\nSuperclass Macro-Averaged Metrics:")
        print(f"Macro-Precision: {macro_precision_super:.2%}")
        print(f"Macro-Recall: {macro_recall_super:.2%}")
        print(f"Macro-F1-score: {macro_f1_super:.2%}")

    # print class accuracies
    print("\nClass Accuracies:")
    sorted_class_accuracies = sorted(class_accuracies, key=lambda x: x['class'])
    for item in sorted_class_accuracies:
        tabledata.append([item['class'], item['accuracy']])

    max_len = max([len(row[0]) for row in tabledata])
    for row in tabledata:
        print(f"{row[0]:<{max_len}}: {row[1]:6.2%}")

    if(num_classes == 100):
        # Calculate the average superclass accuracy
        average_class_accuracy_100 = calculate_average_accuracy(class_accuracies)

        # Print the average superclass accuracy
        print(f"\nAverage class Accuracy: {average_class_accuracy_100:.2%}")    

    if(num_classes == 20):
        # Calculate the average superclass accuracy
        average_superclass_accuracy_20 = calculate_average_accuracy(class_accuracies)

        # Print the average superclass accuracy
        print(f"\nAverage Superclass Accuracy: {average_superclass_accuracy_20:.2%}")

    if(num_classes == 100):
        class_metrics_list = class_metrics_list_sub
        
    # Print precision, recall, and F1-score for each super-class
    print("\nPer-Class Metrics:")
    sorted_class_metrics_list = sorted(class_metrics_list, key=lambda x: x[0])
    for item in sorted_class_metrics_list:
        tabledata_metrics_sub.append([item[0], item[4], item[5], item[6]])

    max_len = max([len(row[0]) for row in tabledata_metrics_sub])
    for row in tabledata_metrics_sub:
        print(f"{row[0]:<{max_len}}: Precision={row[1]:6.2%}, Recall={row[2]:6.2%}, F1-score={row[3]:6.2%}")

    if(num_classes == 20):
        # Print macro-averaged precision, recall, and F1-score
        print("\nclass Macro-Averaged Metrics:")
        print(f"Macro-Precision: {macro_precision_super:.2%}")
        print(f"Macro-Recall: {macro_recall_super:.2%}")
        print(f"Macro-F1-score: {macro_f1_super:.2%}")

    if(num_classes == 100):
        # Print macro-averaged precision, recall, and F1-score
        print("\nclass Macro-Averaged Metrics:")
        print(f"Macro-Precision: {macro_precision_sub:.2%}")
        print(f"Macro-Recall: {macro_recall_sub:.2%}")
        print(f"Macro-F1-score: {macro_f1_sub:.2%}")

    # create the confusion matrix
    confusion_matrix = calculate_confusion_matrix(all_true_labels, all_predicted_labels, num_classes)
    return confusion_matrix, average_superclass_accuracy_20, average_superclass_accuracy_100

if __name__ == '__main__':
    '''
    Main function to test the trained model on the CIFAR100 dataset.
    
    Usage:
    python test.py -f1 <filename1> -f2 <filename2> -d
    
    Example:
    python test.py -f1 model_cnn_20240401-1754_eps-15_lr-5_nc-20
    
    '''
    confusion_matrix_20 = None
    confusion_matrix_100 = None
    average_superclass_accuracy_20 = None
    average_superclass_accuracy_100 = None
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test trained model on CIFAR100 dataset.')
    # add the first model's filename
    parser.add_argument('-f1', '--filename1', type=str, help='Name of the first model file to load')
    # add the second model's filename
    parser.add_argument('-f2', '--filename2', type=str, help='Name of the second model file to load')
    # add the debug flag
    parser.add_argument('-d' , '--debug', action='store_true', help='Enable debug mode by correcting directory paths.')
    args = parser.parse_args()

    fp = []    
    
    # Append filenames to be used for testing
    if args.filename1 is None:
        fp.append(None)
    else:
        fp.append(args.filename1)
    
    if args.filename2 is None:
        fp.append(None)
    else:
        fp.append(args.filename2)

    for i in range(2):
        
        # Check if the filename is provided
        if (fp[i] is None):
            continue
        # Load the trained model using the provided filename
        model, model_type, num_classes = load_model(fp[i], args.debug)

        if model is not None:
            # Load the test dataset
            transform = transforms.Compose([transforms.ToTensor()])
            if (args.debug):
                testset = CIFAR100(root='./code/data', train=False, download=True, transform=transform)
            else:
                testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
            
            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

            # Move the model to the appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            # Define the criterion (loss function)
            criterion = torch.nn.CrossEntropyLoss()
            
            confusion_matrix, average_superclass_accuracy_20_temp, average_superclass_accuracy_100_temp = evaluate(model, model_type, num_classes, testloader, device, criterion)
            if (num_classes == 20):
                confusion_matrix_20 = confusion_matrix
                average_superclass_accuracy_20 = average_superclass_accuracy_20_temp
            if (num_classes == 100):
                confusion_matrix_100 = confusion_matrix
                average_superclass_accuracy_100 = average_superclass_accuracy_100_temp

    # Plot the available confusion matrix
    if confusion_matrix_20 is not None and confusion_matrix_100 is None:
        # Print the average superclass accuracy for a model trained on 20
        print(f"\nAverage Superclass Accuracy of a model trained on 20 classes: {average_superclass_accuracy_20:.2%}")

        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix_20, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix (20 Classes)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    if confusion_matrix_100 is not None and confusion_matrix_20 is None:
        # Print the average superclass accuracy for a model trained on 100
        print(f"\nAverage Superclass Accuracy of a model trained on 100 classes: {average_superclass_accuracy_100:.2%}")

        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix_100, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix (100 Classes)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    if confusion_matrix_100 is not None and confusion_matrix_20 is not None:
        # Print the average superclass accuracy for a model trained on 20
        print(f"\nAverage Superclass Accuracy of a model trained on 20 classes: {average_superclass_accuracy_20:.2%}")
        # Print the average superclass accuracy for a model trained on 100
        print(f"Average Superclass Accuracy of a model trained on 100 classes: {average_superclass_accuracy_100:.2%}")

        # Plot both confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].imshow(confusion_matrix_20, cmap='viridis', interpolation='nearest')
        axes[0].set_title('Confusion Matrix (20 Classes)')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        axes[1].imshow(confusion_matrix_100, cmap='viridis', interpolation='nearest')
        axes[1].set_title('Confusion Matrix (100 Classes)')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

        plt.tight_layout()
        plt.show()
