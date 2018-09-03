In this file I outline an overview of my code followed by a description of my general approach to 
this problem, including some ideas that didn't make it to the final submission.

## Overview

### The Model

I used pre-trained convolutional layers from VGG11 as a feature extractor, freezing the weights and
appending a fully connected network as the actual speed predictor. However, speed is not easily
inferred from a single image frame, so instead each of the training examples was a bundle of 3
frames - the frame that we want to infer the speed for, the previous frame, and the following frame.
Each of these frames was fed into its own instance of the VGG11 convolutional layers, and the final
outputs from each of these convolution sequences were fed into a bidirectional GRU, in order to
better capture the relationship of speed with consecutive frames. These outputs were concatenated 
and fed into fully connected layers that output the speed prediction.

### Results

After training for 20 epochs, my model ended up with 2.05 training mean-squared error and 18.82
validation mean-squared error (10% of data set aside for validation). This high discrepancy between 
training and validation MSE is indicative of a great amount of overfitting, and though I tried to 
combat this by shrinking the frame width and height by a factor of 4 as well as adding dropout and 
L2 regularization, being limited to the 3-channel VGG11 input (due to loading pre-trained weights) 
is unfortunate because I think that grayscale frames would greatly contribute to overfit reduction, 
perhaps at the cost of some additional training loss.

I also would have liked to put the rest of the validation data into the model and train with the
same parameters, but my GPU ran out of memory for just a single epoch when adding the final 10% of
data back in, so I left the model as is.

### Organization

- **model.py** - model definition and training/inference script (structured a bit weirdly, see 
**trainer.py** for more details)
- **dataset.py** - dataset definition
- **trainer.py** - script to run in order to train the model. This is required due to my PC's memory
limitations. Since I only have enough GPU memory to train one epoch per script call, I structure
the training process as a sequence of script calls and structure **model.py** accordingly.
- **optflow.py** - optical flow dataset feature extraction experiments (not used in final model)
- **data/** - contains test and training videos as well as training labels and test predictions
- **models/** - contains my final model

### Challenges

The main challenge I faced was limitation in computational resources. There were a lot of design
decisions that I was forced to make due to my GPU (GTX965M) not having enough memory to handle even 
a single epoch of training the model. I also had to structure my training script in a way where each
epoch was saved so that I could use an automated script to end the instance (freeing up memory) and
start an additional epoch after loading the previous one.

## My Approach

### Step 1: Optical Flow

First I tried using an optical flow approach by checking correspondences of feature points from
frame to frame, first using a Shi-Tomasi corner detector to acquire the points, then using the
Lucas-Kanade optical flow algorithm to identify potential correspondences. However, this method
proved to be unsuccessful due to having trouble identifying correspondence points and therefore
the resulting features were pretty sparse and incapable of providing enough information. The code
for my initial optical flow feature extraction is in optflow.py (none of the code is used in the
final submission) for reference.

### Step 2: Convolutional Neural Networks

The logical next step was to turn to CNNs, though I was aware of my computing limitations, so
training my own custom model was out of the question. Instead, I was set on leveraging the
convolutional layers of an existing architecture as a feature extractor, keeping those weights
frozen as I trained my own GRU + fully connected layers at the end of the network to serve as the 
speed predictor itself. I chose VGG11 because this seemed like a good balance of model quality 
versus number of parameters my computer could train in a reasonable amount of time.

One of the issues I faced was being forced to conform to VGG11's specific input/output scheme, since
it only accepted single RGB images (I had originally intended to use grayscale. Because I needed to
use the existing feature extractor, I decided to brute-force it, using RGB images and concatenating
the feature extraction results to funnel into the fully connected layers.

I initially made the incorrect assumption that I might be better off using fewer VGG layers, which
might save me time and processing power, but in fact the pooling layers at the end of every VGG
convolutional block made it much more feasible to use the full VGG11 network. Using just the first
VGG layers resulted in an output volume that had gone through less pooling, therefore resulting in
some especially high parameter counts for the fully connected layers at the end.

### Step 3: Optimization Decisions

In training this network I chose to not shuffle before the training-validation data split. This is
because, due to each training point being a bundle of 3 frames, this training-validation data
sharing would likely happen a significant amount. I also considered making the bundle 5 frames to
get more data for cases where 3 frame bundles could have trouble determining the speed, such as when
the car passes under a tunnel, but decided against it in the end due to processing power and 
overfitting concerns.

Also, because running on CPU was too slow and I only had enough memory on GPU to run one epoch, I
had to write a script to continuously run python scripts for each epoch, loading the previous epoch
and saving a new one.

I used Adam as a gradient descent algorithm and though that is perhaps not ideal, I did not have the
time to tune the SGD+momentum hyperparameters to their optimal values.