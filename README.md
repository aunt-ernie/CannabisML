Decoding Cannabis Use from Structural and Diffusion MRI

Features
sMRI features were derived from Mindboggle (Freesurfer + ANTs) (Klein et al., 2017)

dMRI features were derived from TBSS+MELODIC (FSL) (Schouten et al., 2017)

Feature Normalization
A GLM used to determine residuals, controlled for covariates. Residuals were subsequently used as features during training/testing.

𝑦̂ = 𝛽0 + 𝛽1𝐼𝐶𝑉 + 𝛽2𝑆𝑒𝑥 + 𝛽3𝐴𝑔𝑒 + 𝛽4𝑆𝑖𝑡𝑒 

𝜀 = 𝑦 − 𝑦̂  

𝑟𝑒𝑠𝑖𝑑𝑢𝑎𝑙𝑠 = 𝜀 

Feature Selection
An ANOVA was used to determine F-stastics and the top  𝑛𝑡ℎ  percentile of F-statistics was during training/testing.

The 𝑛𝑡ℎ percentile was determined via gridsearch.

SVM and DNN
Nested, statified k-folds cross validation was used for both the SVM and DNN classifier.

SVM hyperparamters include cost (𝐶), gamma (𝛾), and kernel (𝜅).

DNN hyperparameters inlcude initalizers, activation functions, batch size and weight decay.

Both classifiers include ANOVA percentile as a hyperparameter.

DNN archtecture include two hidden layers, the number of nodes in these layers was determined following suggestions given by Huang (2003). The network had one output layer with a binary cross-entropy activation function, required for binary classification tasks.

Cross-validation, grid-search, and SVM was implemented using sci-kit learn (Pedregosa et al., 2011) and the DNN was implemented using Keras (Chollet, 2015).

References
Chollet, F. (2015) Keras, GitHub. https://github.com/fchollet/keras.

Huang, G.B. (2003). Learning capability and storage capacity of two-hidden-layer feedforward networks. IEEE Transactions on Neural Networks, 14, 274–281. doi: 10.1109/TNN.2003.809401

Klein, A., Ghosh, S. S., Bao, F. S., Giard, J., Häme, Y., Stavsky, E., … Keshavan, A. (2017). Mindboggling morphometry of human brains. PLOS Computational Biology, 13(2). doi: 10.1371/journal.pcbi.1005350

Pedregosa F., Varoquaux, G., Gramfort, A., Michel V., Thirion, B., Grisel, O... Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 825-2830.

Schouten, T. M., Koini, M., Vos, F. D., Seiler, S., Rooij, M. D., Lechner, A., … Rombouts, S. A. (2017). Individual classification of Alzheimers disease with diffusion magnetic resonance imaging. NeuroImage, 152, 476–481. doi: 10.1016/j.neuroimage.2017.03.025
