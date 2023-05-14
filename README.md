# Android malware detection system relies on binary features backdoor attacks

### Dependencies
The project requires the following Python libraries:

- TensorFlow
- TensorFlow Federated
- scikit-learn
- pandas
- numpy
- scikit-learn


### Data
The dataset is composed of Android applications binary features extracted from the AndroZoo repository, specifically those applications that were published in the year 2018 and onwards. Each feature in the dataset represents a particular characteristic of an Android application, and a target variable indicates whether the application is benign (0) or malicious (1).



### Preprocessing
Before running the federated learning scenario, the data is split into a training set and a test set. The training set is further divided into smaller subsets, each representing a local dataset on a client device.

### Model
The machine learning model used in this project is a simple feedforward neural network created using TensorFlow's Keras API. The model consists of an input layer, a hidden layer with 100 neurons and a ReLU activation function, and an output layer with a single neuron and a sigmoid activation function.



### Federated Learning
Federated learning is implemented using TensorFlow Federated (TFF). The training process consists of several rounds. In each round, the current global model is sent to each client. Each client trains the model on their local data and sends the model updates back to the server. The server then averages these updates to create a new global model.
