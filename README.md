# Flowers_Image_Classifier_Project
<h2>Development Notebook</h2>
<ul>
  <li>Package Imports - All the necessary packages and modules are imported in the first cell of the notebook</li>
  <li>Training data augmentation - torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping</li>
  <li>Data normalization - The training, validation, and testing data is appropriately cropped and normalized</li>
  <li>Data loading - The data for each set (train, validation, test) is loaded with torchvision's ImageFolder</li>
  <li>Data batching - The data for each set is loaded with torchvision's DataLoader</li>
  <li>Pretrained Network - A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen</li>
  <li>Feedforward Classifier - A new feedforward network is defined for use as a classifier using the features as input</li>
  <li>Training the network - The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static</li>
  <li>Validation Loss and Accuracy - During training, the validation loss and accuracy are displayed</li>
  <li>Testing Accuracy - The network's accuracy is measured on the test data</li>
  <li>Saving the model - The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary</li>
  <li>Loading checkpoints - There is a function that successfully loads a checkpoint and rebuilds the model</li>
  <li>Image Processing - The process_image function successfully converts a PIL image into an object that can be used as input to a trained model</li>
  <li>Class Prediction - The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image</li>
  <li>Sanity Checking with matplotlib - A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names</li>
</ul>

<h2>Command Line Application</h2>
<ul>
  <li>Training a network - train.py successfully trains a new network on a dataset of images</li>
  <li>Training validation log - The training loss, validation loss, and validation accuracy are printed out as a network trains</li>
  <li>Model architecture - The training script allows users to choose from at least two different architectures available from torchvision.models</li>
  <li>Model hyperparameters - The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs</li>
  <li>Training with GPU - The training script allows users to choose training the model on a GPU</li>
  <li>Predicting classes - The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and its associated probability</li>
  <li>Top K classes - The predict.py script allows users to print out the top K classes along with associated probabilities</li>
  <li>Displaying class names - The predict.py script allows users to load a JSON file that maps the class values to other category names</li>
  <li>Predicting with GPU - The predict.py script allows users to use the GPU to calculate the predictions</li>
</ul>
