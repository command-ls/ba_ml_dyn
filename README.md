Code, dataset, model and trajectory used in "Construction of Highly Accurate Machine Learning Potential Energy Surfaces for Excited-State Dynamics Simulations Based on Low-Level Dataset", by Shuai Li, Bin-Bin Xie, Bo-Wen Yin, Lihong Liu, Lin Shen, and Wei-Hai Fang.

Here is an introduction to the contents of the four directories:

**The first directory (code)** contains the BPNN and SchNet model architectures we use, as well as the code needed to run the dynamics. Note that the BPNN model is only used for ethylene-bridged azobenzene (EBA); the SchNet model is only applicable to organic molecules and cannot handle periodic samples. The dynamics is based on surface hopping theory combined with energy based coherence, and the system velocity is adjusted along the NAC direction.

**The second directory (datasets)** contains 5 datasets, namely training set, validation set, test set, external test set A, and external test set B. Except for external test set A, which contains only geometries and energies, all other datasets contain geometries, energies, and forces. Energy and force are stored in numpy format. Geometry is stored in xyz format. The geometry of external test set A consists of 39 parts, each extracted from the CASSCF+AIMS dynamics trajectory. Note that the data format required by non-neural network models is inconsistent with BPNN. Taking descriptors as an example, the input form for non-neural network models is (number of samples, atomic number multiplied by descriptor dimension), while the required input form for BPNN is (number of atoms, number of samples, descriptor dimension). In addition, the SchNet model inputs the ASE database, so we need to store information such as geometries, energies, and forces in the database in advance.

**The third directory (trajectories)** contains the initial conditions, OM2/MRCI dynamics trajectories, and machine learning (ML) dynamics trajectories of three ensemble models. The initial conditions are generated by MOLPRO software based on Wigner sampling, and the resulting structure and momentum are in atomic units. We have prepared 200 initial conditions for both E and Z directions. Each trajectory retains only structure, energy, and state.

**The fourth directory (tuned models)** contains the machine learning tuned models. The non-neural network models include KRR, SVM, and GBDT models, which store the best models. Neural network models include BPNN and SchNet, which store the tuned models and the mean and standard deviation used in training.
