# Gravitational_Wave_Detection
Used supervised machine learning to detect the presence of a signal from time series data containing simulated gravitational wave measurements from a network of 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo).

The project was completed as part of a course project for the course CS235: Data Mining Techniques in Fall 2021.
The dataset was part of a kaggle competition https://www.kaggle.com/c/g2net-gravitational-wave-detection. Each data sample consists of 3 time series of interval 2 seconds sampled at 2048 Hz and hence was stored as [3 X 4096] array. 

A sample with signal present (labeled y = 1) 

![label0](label_1.png)

compared to a sample with label y = 0 (only background noise)

![label1](label_0.png)

shows that no signs of signal jumps out of the naked eye.


Bandpass filter was applied in the range [20,500] Hz based on the knowledge that gravitational wave signals from binary black hole mergers generally appear in the range [20,350] Hz. Following is a sample data containing a signal event after bandpass filtering

![filter](filter.png)

Next the time series data was converted to the frequency domain using constant transorm and Short-time Fourier transform which gave a slightly better glance at the signal.

<img src="CQT.png" alt="drawing" height= "400" width="500"/>

Apart from this, other transformations like Gramian Angular Fields and Markov transition fields are also tried out. https://arxiv.org/pdf/1506.00327.pdf



| [![GAF](GAF.png) Gramian Angular Fields  | [![MTF](MTF.png) Markov Transition Fields | 
|:---:|:---:|
                                                        

