# Applications of Connectivity in the Human Brain Using Functional MRI data: Classifcation with Graph Neural Networks

_This is our DSC 180B Capstone Project (Section A09):_ Using graph neural networkds to classify gender and age based on connected networks in the brain is new in neuroscience. 

For our second quarter project, we plan to expand on findings from our first quarter project where we used more statistical-based methods. We plan to implement a neural network-based solution to find resting state brain networks in order to classify features such as gender and age. The resting state networks are interesting because it is highly active when a brain is comprehending something or thinking about the self. We want to discover if we can use this fact to differeniate male and female or young and old brains. To see this network in fMRI data, the patient is asked to think about nothing. Overall our project is engaged in the mysteries of our brain which cannot only be solved by statistical approaches. Neural networks are beneficial for finding relationships in the data that would not otherwise be easily discovered. Once we have a trained model, we are able to release the architecture and trained parameters of the neural network so our work will be easily reproducible.

## Project Structure

```
|____README.md
|____requirements.txt
|____Notebooks
  |____HCP_Data.ipynb
  |____Andrew_Analysis.ipynb
  |____eda.ipynb
  |____sim_correlation_gen.ipynb
  |____GNN.ipynb
|____Data
  |____data_clean.csv.gz
  |____HCP
  |____subject_Andrew
```

## Accessing and Retriving Data

 * Subject Data is sourced from the Human Connectome Project ([HCP](https://www.humanconnectome.org/study/hcp-young-adult)). The data is directly downloaded from their [database](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=A179A9A1C462DE17533CA546A12C0485). The necessary data has already being downloaded and stored in this notebook. The HCP Notebook merges the columns of interest in the HCP raw data and generates the cleaned data for other notebooks to access.
 * Subject Andrew, which is one of the capstone team member, underwent and fMRI brain scan at UCSD. His data was given to us from the Center for functional MRI and run through the MRI preprossing pipeline to get the raw data of the brain scan in the NIFTI 4D file format.

## Running the Project

 * To install the python packages, run the following command from the root directory of the project: `pip install -r requirements.txt`
 * Open up any of the juypter notebooks at the section you want to reproduce

## Reference

* [Hand Book of Functional MRI Data Analysis](https://www.cs.mtsu.edu/~xyang/fMRIHandBook.pdf)
