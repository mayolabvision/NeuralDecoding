### Data format 

#### Filename:
e.g. **MTFEF-pa29dir4A-1600ms.pickle**
* "MTFEF" = what brain regions neurons are recorded from (MT and FEF)
* "p" = who recorded the data (Patrick)
* "a" = first letter of monkey's name (Aristotle)
* "dir4" = how many motion directions there are in this task (4 orthogonal directions)
* "A" = which session of the day is this (session A)
* "1600ms" = from each trial, how many time points are we looking at? (1600ms = 800ms before target starts to move + 800ms after target starts to move) 

[What is a .pickle file?](https://www.datacamp.com/tutorial/pickle-python-tutorial)

#### What is contained in the .pickle file?:
Neuronal information (inputs) -- binned spike rates for each neuron:
* **spike_times**: "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
Behavioral information (outputs) -- binned features:
* **vels**: "number of time bins" x 2 (x and y velocities), where each entry is the eye velocity in a given time bin 
* **pos**: "number of time bins" x 2 (x and y positions), where each entry is the eye position in a given time bin 
* **vel_times**: "number of time bins" x 1, vector that states the time at all recorded time points

The pickle file is generated with the *mat_to_pickle.py* file in the handy\_functions/ directory, which takes in a preprocessed .mat file.
