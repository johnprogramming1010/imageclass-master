TITLE OF PROJECT

Project Title: Enhanced Image Classification Android App

Our project entails the development of an innovative Android application leveraging computer vision to classify images uploaded within the app. Utilizing Android Studio as our development environment, we employ a blend of Java, XML, and Python to craft this intuitive tool.

Key Components:

1. Java: We harness the power of Java to execute the backend operations, specifically to discern and present the class and sub-class of the uploaded images.
2. XML: XML serves as our design language, enabling us to tailor the app's interface to meet our unique specifications. Through XML, we ensure a seamless user experience, enhancing usability and visual appeal.
3. Python: Our computer vision algorithms are scripted in Python, leveraging its robust libraries and frameworks. With the integration of PyTorch, a cutting-edge extension, we empower our app to learn and classify images effectively. PyTorch facilitates training on CIFAR-100, a comprehensive dataset comprising 100 diverse sample images, enabling accurate classification.
4. Tensors: Tensors, versatile arrays capable of holding various data types, are instrumental in our image classification pipeline. PyTorch's tensor support enhances efficiency and flexibility in handling image data, contributing to the app's robust performance.

Project Objective:

The primary objective of our endeavor is to empower users with an intuitive tool for image identification and classification. By leveraging computer vision technologies, we aim to simplify the process of recognizing and categorizing images. Ultimately, our app strives to enhance user experiences, facilitating better understanding and interpretation of visual content.


*********************************************
*********** Using the Application ***********
*********************************************

1. Begin by downloading the "app name" onto your Android device from the designated app store.
2. Upon launching the app, tap on "Load Image" to select an image you wish to classify using the app.
3. Once you've chosen an image, you'll be seamlessly redirected to the home page, where your selected picture will be prominently displayed.
4. To initiate the classification process, simply tap on "Classify". Sit back and let the app determine the image's name and assign it to a super class category for easy identification and organization.

***** How to download and use the app on android studio ******

1. Start by downloading Android Studio. You can find detailed installation instructions at https://developer.android.com/studio/install.
2. Once Android Studio is installed, head over to GitHub and download the code as a zip file from the following link: https://github.com/SeniorSem/imageclass/tree/master.
3. Open Android Studio and import the downloaded project. This process might take a few minutes as Android Studio loads the project files, so please be patient.
4. Upon successful loading of the code within Android Studio, navigate to the device manager to select an Android device to run the app on.
5. With your chosen device selected, hit the "Run" button within Android Studio. Sit back and allow some time for the app to load onto the device. This process may take a while, so patience is key.
6. Once the app is loaded on your device, tap on "Load Image" to select an image you wish to classify using the app.
7. Upon selecting an image, you'll be redirected to the home page where your chosen picture will be displayed. To proceed with classification, tap on "Classify". The app will then provide the image's name along with its assigned super class category for easy identification.



******* Collaborators ********
Jeremy Scott, John James, Himanshu Patel, Theodore Carter


*********************************************
****************** Models *******************
*********************************************

>> CNN

>> Linear

*********************************************
****************** Graphs *******************
*********************************************



*********************************************
*************** Coding Files ****************
*********************************************

>> SeeCats.py

To be deleted

>> app_model.py:

Purpose: Converts .pt file to a file usable on android devices

Arguments:
- File
- Output

To use this file, use two arguments, one is the input .pt file name and then the output filename.  The program will adapt the given model into an model that can be used on android devices

>> generate_cifar_images.py

Purpose: converts binary images files to PNG files

Arguments:
- File

When running this file, you need to use an argument to select the images to be received.  The CIFAR-100 returns several files when unpacked including two files called 'test' and 'train'

The images stored in 'train' are the files used when training the CIFAR-100 model
The images stored in 'test' are the files that are used when testing the models.

These are different so the model is not tested against memorized images.  Only new images should be used in testing.

>> train.py

  model_cnn.py

  model_linear.py

>> test.py

*********************************************
*************** Android Files ***************
*********************************************

ON GOING ISSUE!!!
When loading the application for the first time, gradle is not properly loading pytorch into android studio.
CURRENT FIX
- Under gradle scripts select build.gradle.kts (Module :app)
- Find two lines labeled ('implementation ("org.pytorch:pytorch_android:2.1.0")', 'implementation ("org.pytorch:pytorch_android_torchvision:2.1.0")')
- Change version to 2.0.0 and sync the project at the top right of the screen
- Change version back to 2.1.0 and sync the project once again
- Check the MainActivity.java file and see if pytorch imports are working.


