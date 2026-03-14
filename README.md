## Model Architecture

This project implements a structure-based deep learning pipeline for ligand–protein binding prediction.

The system consists of two related models:

### 1. Bilinear Affinity Model

The primary model learns interaction patterns between ligand features and protein pocket residues.

Input representation:

- **Ligand features (11D)**  
  Physicochemical descriptors computed from molecular structures (RDKit).

- **Protein pocket features (37D per residue)**  
  - 20D amino-acid one-hot encoding  
  - 13D physicochemical residue properties  
  - 4D geometric features (ligand–residue relative position and distance)

Model components:

- Ligand feature encoder
- Residue feature encoder
- Low-rank bilinear interaction layer
- Attention-based residue pooling
- Affinity regression head

Training dataset:

- **PDBbind refined set**

Evaluation:

- **CASF-2016 benchmark set**

This model learns interaction weights that capture ligand–pocket binding patterns.


### 2. AttnBind Classification Model

A second model was built on top of the trained bilinear model to estimate binding probability.

The AttnBind model:

- Initializes its parameters from the trained bilinear affinity model
- Applies attention-based refinement over pocket residues
- Predicts binding probability

Additional outputs include:

- residue attention weights
- interaction contribution heatmaps

### Model Workflow
Ligand structure (SDF / DrugBank)
│
Ligand feature extraction (RDKit)
│
Protein pocket feature extraction (PDB)
│
Bilinear Affinity Model
(training on PDBbind)
│
CASF-2016 evaluation
│
AttnBind classification model
│
DrugBank screening & interaction analysis

![Workflow](workflow.png)


### Application

The trained models were used for **drug repurposing screening**, evaluating potential binding interactions between DrugBank compounds and target protein pockets.
