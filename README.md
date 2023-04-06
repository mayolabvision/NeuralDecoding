# Decoding of eye movement trajectories by neuronal populations in cortex
In this project, we are using machine learning methods to investigate the relationship between cortical activity and eye movements in macaque monkeys. We recorded from a small population of neurons in two regions of the brain known to play a role in eye movement generation, the Frontal Eye Fields (FEF) and the Medial Temporal (MT) area, while a monkey kept their eyes fixed on a moving target. Eye movements were recorded using an infrared eye tracker. The main goal of this work is to reconstruct where the monkey's eyes were looking on a moment-to-moment basis based on the spiking activity of FEF and MT neurons. 

This repository contains python jupyter notebooks (with various decoding and machine learning methods) and datasets (with eye positions/velocities and neuronal spiking activity). The packages used in the python notebooks are from the Kording Lab's Neural Decoding package ([GitHub Link](https://github.com/KordingLab/Neural_Decoding)) and the data was collected by Dr. Patrick Mayo during his postdoc at Duke University. 

## Project Information:
### Motivation: 
In our lab, we are interested in understanding how the brain coordinates eye movements. We move our eyes constantly, around ~3 times per second, to position items of interest in the world onto the high acuity portion of our retinas (i.e., the fovea). This is something we do effortlessly, but the brain circuitry in control of visual processing and eye movements is complex. 
As an example, imagine you are trying to swat a fly. As the fly zooms around the room, you are trying to keep your eyes fixated on its flight trajectory so when it eventually lands you can be ready to pounce. In your brain, there are different regions responsible for handling different aspects of this scenario. There are *sensory* regions that keep track of where the fly is and how fast it is moving; and then there are *motor* regions which determine where you'll need to look based on the information sent to it by the sensory areas. (And there are cognitive regions that make predictions about the fly based on experience, etc.) To accurately keep your eyes positioned on the fly, these brain regions with seemingly different roles have to rapidly communicate back and forth.
The coordination of sensorimotor signals occurs at the level of neurons, which act as information messengers via electrical impulses. Neurons within one brain region and across different brain regions communicate with one another to accomplish some goal, like moving the eyes to track a fly. But even within a single brain region, neurons may respond to different things and may be sensitive to different aspects of our environments. Some neurons may care about the identity of a visual stimulus (e.g. fly, bee, or butterfly) whereas other neurons might be sensitive to the direction of flight. 

It is still unknown how these neurons, that are selective to different features of the visual world, come together to accomplish natural visual behavior such as adaptively moving the eyes. How can we determine what neurons "care about", and how does neuronal activity change in response to different types of stimuli?

<img align="right" img width="514" alt="neuralDecoding_graphicalAbstract" src="https://user-images.githubusercontent.com/37158560/226387792-1016170b-3429-48a5-9fdc-3a38239a1a6d.png">

### Immediate goals and questions:

1. Can we decode eye movement trajectories from neuronal activity?
    - Previous results (from Bing) show that we can decode what direction the eyes moved from neural activity.
    - What do neurons "care about"? What are they sensitive to? 
2. How do neurons from different brain regions contribute to generating eye movements? 
    - Will neuronal activity in one brain region (FEF v. MT) do a better job of decoding eye movements?
3. How many neurons in each brain region are required to reach some fixed level of decoding performance? 
4. Does the neural decoder generalize well across different stimulus features? How tolerant or intolerant are neurons to specific visual parameters?
    - If you build a decoder based on neural activity recorded when the monkey sees a stimulus with a high contrast, will it perform as well when shown brain activity recorded with a dimmer contrast?
    - What about the speed of a moving target on the screen?
5. How does model efficiency and performance compare across different types of decoders?
    - There are many types of regression (Wiener filter, Wiener cascade, Kalman filter, Naive Bayes, Support Vector Regression, XGBoost, Dense Neural Network, Recurrent Neural Net, GRU, LSTM) and classification (Logistic Regression, Support Vector Classification, XGBoost, Dense Neural Network, Recurrent Neural Net, GRU, LSTM) models to try.

### Future ideas and applications:
1. Can we decode what type of eye movements the monkey is making?
    - From neurons in FEF, which respond to either/both saccades or pursuit, can we decode which type of eye movement they are making? Using *pursuity* neurons, can we decode what direction the monkey pursued a target? What direction they made a saccade? 
2. Can we use this decoding model for real-time, online BCI experiments?
    - As the monkey moves their eyes, could you change the task they are doing based on real-time decoding?
3. Can we use this decoding model to conduct "blind" recording and decodings?
    - Based on the decoding technique that we know works (eye movements from neurons), could you move the recording electrode to an entirely new place blindly and be able to decode anything?
4. Can we use brain-decoded eye positions to inform/complement standard eye movement recordings?
    - Do neural recordings do a good job pairing up to eye tracking data?
    - Could you skip over eye tracking by using a decoder?
5. Could we use these types of decoding models to inform the design of visual prostheses?

### Basic definitions:
* **Neuron**: a specialized cell transmitting nerve impulses
* **Machine Learning**: enables researchers to discover statistical patterns in large datasets to solve a wide variety of tasks, including in neuroscience
* **Decoding**: the act, process, or result of extracting meaning or usable information 
* **Fixation**: the maintaining of the gaze on a single location
* **Spiking activity**: in behavioral neuroscience, a train of electrical signals recorded from an individual neuron in the brain. Spikes are the action potentials or signals generated by neurons to communicate with one another
* **Sensorimotor network**: responsible for sensing physical inputs, converting them to electrical signals that travel throughout the brain network, and then initiating a physical response
* **Saccade**: a rapid movement of the eye between fixation points
* **Smooth pursuit**: a type of eye movement in which the eyes remain fixated on a moving object


## Getting Started:

###  Installing all necessary code and packages
**Modules you will need**:
* python
* homebrew
* pip
* git
* Jupyter
* TensorFlow
* Keras
* XGBoost
* scikit-learn
* BayesianOptimization

#### For Mac OS X
Open the terminal and enter the following commands to ensure you have all of the necessary programs/packages installed:

1. Download or clone this repository (mayolabvision: NeuralDecoding) onto your local machine.
```buildoutcfg
# Change the current working directory to the location where you want the cloned directory.
$ git clone git@github.com:mayolabvision/NeuralDecoding.git

# If it says "Unpacking objects: 100%... done., then it worked!
```
In this project-dedicated repository, you will find:
* **Glaser_Chowdhury-Machine_Learning_for_Neural_Decoding-2020.pdf**: paper with more information about how each type of decoder works and how they are applied to neuroscience
* **example/decoding_eye_position_example.ipynb**: example python jupyter notebook showing how to upload the preprocessed data (type .pickle) and run each of the decoders
* **example/results/**: outputs from example python notebook 
* **/datasets **: contains pre-processed .pickle files for each recording session (explained in more detail in *data format* section)
* **/handy_functions**: contains pre-processing functions (both in python and matlab), code for the decoding models, and other metrics.     

2. Install **Homebrew**
```buildoutcfg
# Check if you already have homebrew
$ which brew

# If you see /usr/local/bin/brew, you already have it! If you don't...
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Enter your macOS credentials if and when asked. If you get a pop-up alert asking you to install Apple's command line developer tools, install them and homebrew will finish installing after. 

3. Install **Python3**

```buildoutcfg
# Check if you already have python
$ python --version

# If you see 'Python 3.9.5', you already have it! If you don't...
$ brew update && brew upgrade
$ brew install python3

# To check if this was successful...
$ pip3

# You should only see the help information from pip3 if your python installation was successful
```

4. Install the **Neural_Decoding** package from the Kording Lab.

```buildoutcfg
$ pip install Neural-Decoding
```
This should install of the basic packages and dependencies you will need to run all of the decoders, but if it doesn't ask Kendra for help.


5. Install **Jupyter Notebook**

```buildoutcfg
$ pip install notebook
```

6. Install the **Neural_Decoding** package from the Kording Lab.

```buildoutcfg
$ pip install Neural-Decoding
```

#### For Windows

1. Download **Anaconda**.  
Python and all the necessary packages come with Anaconda, so you do not need to install them separately.


2. Launch **Jupyter Notebook** from Anaconda Navigator


3. Click New - Folder from the top right


4. Ceate python file within the folder


5. Install the **Neural_Decoding** package from the Kording Lab.
```buildoutcfg
# Type the command below to the notebook
$ ! pip install Neural-Decoding
```
This should install of the basic packages and dependencies you will need to run all of the decoders, but if it doesn't ask Kendra for help.



## Other Useful Links:

* [Reading .mat files in Python](https://discourse.julialang.org/t/how-to-read-tables-out-of-a-mat-file/70098)
* [Using TensorFlow w/ Silicon Macs, via MiniForge](https://towardsdatascience.com/installing-tensorflow-and-jupyter-notebook-on-apple-silicon-macs-d30b14c74a08)
