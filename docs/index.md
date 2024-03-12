# Applications of Connectivity in the Human Brain Using Functional MRI data: Classification with Graph Neural Networks 

| Andrew Cheng | Daphne Fabella| Terho Koivisto | Daniel Zhang | Gabriel Riegner | Armin Schwartzman |
| ---| --- | ----| ----- | ----- | ----- |
apcheng@ucsd.edu | dfabella@ucsd.edu | tkoivisto@ucsd.edu | yiz029@ucsd.edu| gariegner@ucsd.edu | armins@ucsd.edu |

## Sources
[Code](https://github.com/AndrewCheng02/DSC180B-Capstone-ProjectA09)  
[Report](https://www.overleaf.com/project/65c86bd071f00f87c475dce6)  
[Poster](./capstone_poster.pdf)

## Abstract

In our project, we implement a neural network-based application to discern
gender difference from functional magnetic resonance imaging (fMRI) resting
state data. An important region that activates while the brain is at rest is the
default mode network (DMN). While we don’t directly measure the activation
of the DMN, it proves to be a key differentiator for the classification problem.
Our research will walk through steps of statistical methods, simulation, and
model description to achieve our task.

## Introduction

Functional magnetic resonance imaging still proves to be a key tool to deciphering the mysterious architecture of the brain after its first uses in the early nineties. Due to the high spatial resolution (in millimeters) and relative ease of getting a scan, data for fMRI scans has become highly available and useful. These scans still lack temporal resolution due to how fMRI blood-oxygen-level-dependent (BOLD) signals work, where it is the blood flow that is being measured rather than the signals in the brain. This will make our temporal resolution in seconds rather than some other scanning methods (e.g. MEG, EEG, NIRS)  that are in milliseconds. Understanding and simulating fMRI data is crucial for understanding the connectivity of the brain.

Our project will focus on data from the Human Connectome Project (HCP) Young Adult data release, which has a collection of 1,200 fMRI scanned adult brains. The data is in the form of voxels, which are pixels that exist in a three-dimensional space of a brain mapping to an intensity of a BOLD signal at a specific point in time. With the amount of different brains, it would be impossible to map one to one comparisons between specific voxel locations in the brain. So we use pre-processed data that shrinks the number of subjects to 1,003 subjects. We will later discuss the data in more detail. 

We are not the only ones to be interested in the differences of fMRI data in resting state between genders. A study by \cite{Sie2019-np} investigated age and gender differences in resting state functional connectivity. They found during rest females will have stronger correlations in the DMN than their male counterparts. While these differences were shown to wear off with age, this still proves there exist possible indicators to be found in fMRI data.

For our neural network based classifier, we must explain some important considerations. For data pre-processing we explain changes we might need to make to data. Simulate an environment to prove the classification is practicable from correlation data. Finally explain our exploratory data analysis to display significant differences in the data between genders.

## Connectome Figures

- Add Plotting
