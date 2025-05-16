Linear Algebra for Machine Learning and Deep Learning

🟠 Scalars
What are Scalars?
A scalar is a single numeric quantity, fundamental in machine learning for computations, and in deep learning for learning rates and loss values.
➤ Importance: Essential for all computations.


🟢 Vectors
What are Vectors?
Arrays of numbers that can represent multiple forms of data.

In ML: Represent data points.

In DL: Represent features, weights, biases.
➤ Very Important

Row Vector and Column Vector
Different representations of vectors affect computations like matrix multiplication, which is critical in neural network operations.

Distance from Origin
The magnitude of a vector from the origin.

ML: Used in normalization.

DL: Helps understand magnitude of weights/features.
[L] Learn Later

Euclidean Distance
Measures straight-line distance between two points/vectors.

ML: Clustering, k-NN.

DL: Used in loss functions like MSE.

Scalar-Vector Operations
Addition/Subtraction (Shifting): Useful for normalization and bias correction.

Multiplication/Division (Scaling): Used in data scaling and controlling learning rates.

Vector-Vector Operations
Addition/Subtraction: Fundamental for combining/comparing data.

Dot Product:

ML: Similarity measures.

DL: Weighted sums in neural nets.

Angle Between Two Vectors
Indicates directional difference, used in recommender systems and deep vector analysis.

Unit Vectors
Important for normalization and directionally consistent updates in DL.

Projection of a Vector
Used for dimensionality reduction and visualizing high-dimensional features.

Basis Vectors
Define coordinate systems and are used in PCA, SVD, and internal network representation.

Equation of a Line in n-D
ML: Linear regression.

DL: Hyperplanes separating classes.

Vector Norms
Measures vector length.

ML: Regularization.

DL: Batch/Layer normalization.
[L] Learn Later

Linear Independence
Fundamental for:

ML: Regression stability, PCA assumptions.

DL: Reducing redundancy in input space.

Vector Spaces
Spaces for input/output in ML and layer transformations in DL.


🟣 Matrices
What are Matrices?
2D arrays used to represent features, weights, and transformations.

Types of Matrices
Identity: Used in linear ops.

Zero, Sparse: Memory efficient in high-dim data.

Orthogonal Matrices
Used in PCA/SVD. Help prevent vanishing/exploding gradients in DL.

Symmetric Matrices
Equal to their transpose.

ML: Covariance matrices.

Diagonal Matrices
Used in scaling and optimization scheduling.

Matrix Equality
Two matrices are equal if sizes and elements match. Used in convergence checks.

Scalar Operations
Adjust all elements—used in normalization and training updates.

Matrix Addition/Subtraction
Combining datasets or comparing model parameters.

Matrix Multiplication
Central to regression and forward propagation.

Transpose
Required for dot products and specific multiplications.

Determinant
Used in statistical modeling and volume-preserving transformations.

Minor and Cofactor
Foundational for computing inverse and determinants.

Adjoint
Transpose of the cofactor matrix. Used to compute matrix inverse.

Inverse
Used to solve linear equations in ML and pseudo-inverse methods in DL.

Rank
Maximum number of linearly independent rows/columns. Indicates solvability and structure.

Column Space and Null Space
Help determine system solutions.
[L] Learn Later

Change of Basis
Transforms data across coordinate systems.
[L] Learn Later


🔷 Systems of Equations and Transformations
Solving a System of Linear Equations
Used in:

ML: Linear/logistic regression.

DL: Backpropagation as equation-solving.

Linear Transformations
Map input data while preserving relationships. Essential from regression to complex DL models.

3D Linear Transformations
Used in data visualization and geometric understanding.

Matrix Multiplication as Composition
Multiple transformations as one. Important in sequential layers of neural networks.

Linear Transformations with Non-square Matrices
Used in:

Dimensionality reduction.

Feature construction.


🟡 Dot and Cross Product
Dot Product
Results in scalar.

ML: Similarity.

DL: Weighted input sums.

Cross Product
Produces orthogonal vector in 3D space.
[L] Learn Later


🔴 Tensors
What are Tensors?
Generalizations of scalars, vectors, matrices to higher dimensions.

Importance in Deep Learning
Used to represent:

Time series (1D)

Images (2D)

Videos (3D)

Tensor Operations
Addition, multiplication, reshaping for weight/data manipulation.

Data Representation
Images, text, audio—all as tensors in models.


🟤 Eigen Concepts
Eigenvalues and Eigenvectors
Used in:

ML: PCA, transformation analysis.

DL: Optimization behavior.

Eigenfaces
Used in facial recognition.
[L] Learn Later


🟣 Dimensionality Reduction & Factorization
Principal Component Analysis (PCA)
Used for:

Noise removal

High-dimensional visualization

Embedding interpretation
[L] Learn Later

Matrix Factorization Techniques
LU Decomposition – Solves linear systems.

QR Decomposition – Solves regression with stability.

Eigen Decomposition – Finds structure for PCA.

Singular Value Decomposition (SVD) – Used in LSA, dimensionality reduction.
[L] Learn Later
