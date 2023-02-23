# Decoding of eye movement trajectories by neuronal populations in cortex
In this project, we are using machine learning methods to investigate the relationship between cortical activity and eye movements in macaque monkeys. We recorded from a small population of neurons in two regions of the brain known to play a role in eye movement generation, the Frontal Eye Fields (FEF) and the Medial Temporal (MT) area, while a monkey kept their eyes fixated on a moving target. The main goal of this work is to reconstruct where the monkey's eyes were looking on a moment-to-moment basis based on how the spiking activity of FEF and MT neurons. 

This repository contains python jupyter notebooks (with various decoding and machine learning methods) and datasets (with eye positions/velocities and neuronal spiking activity). The packages used in the python notebooks are from the Kording Lab's Neural Decoding package ([GitHub Link](https://github.com/KordingLab/Neural_Decoding)) and the data was collected by Dr. Patrick Mayo during his postdoc at Duke University. 

## Project Information:
### Motivation: 
In our lab we are interested in understanding how the brain coordinates eye movements. We move our eyes constantly, around ~3 times per second, to redirect our attention towards things in our environment that interest us. This is something we do effortlessly, but the brain circuitry in control of visual processing and eye movements is actually extremely complex. 
As an example, imagine you are trying to swat a fly. As the fly zooms around the room, you are trying to keep your eyes fixated on its flight trajectory so when it eventually lands you can be ready to pounce. In your brain, there are different regions responsible for handling different aspects of this scenario. There are *sensory* regions that keep track of where the fly is and how fast it is moving; and then there are *motor* regions which determine where you'll need to look based on the information sent to it by the sensory areas. To accurately keep your eyes positioned on the fly, these brain regions with seemingly different roles have to rapidly communicate back and forth.
This relaying of sensorimotor signals occurs at the level of neurons, which act as information messengers via electrical impulses. Neurons within one brain region and across different brain regions communicate with one another to accomplish some goal, like moving the eyes to track a fly. But even within a single brain region, neurons care about different things and are sensitive to different aspects of our environments. Some neurons may care about the identity of a visual stimulus (e.g. fly, bee, or butterfly) whereas other neurons are sensitive to what direction the flying insect is moving. 

It is still somewhat unknown how these neurons, that are selective to different features of the visual world, come together to accomplish some goal. How can we determine what neurons *care about* and how do their selectivities to certain stimuli modualte their activity?   

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

### Experimental Paradigm:



## Getting Started:
### For Mac OS X
Open the terminal and enter the following commands to ensure you have all of the necessary programs/packages installed:

1. Homebrew
```buildoutcfg
# Check if you already have homebrew
$ which brew
# If you see /usr/local/bin/brew, you already have it! If you don't...
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Enter your macOS credentials if and when asked. If you get a pop-up alert asking you to install Apple's command line developer tools, install them and homebrew will finish installing after. 

2. Python3

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

