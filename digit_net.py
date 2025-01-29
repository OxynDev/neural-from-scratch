import pandas as pd
import random
import json
import numpy as np

def split_data(X, y, train_ratio=0.95):
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    features = features / 255.0
    return features, labels

def generate_weights(
        num_weights: int = 784
    ):
    
    weights_list = []
    for i in range(num_weights):
        weights_list.append(random.uniform(-0.5, 0.5))

    return weights_list

class NeuralNet:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input Layer (784 inputs)
        #     ↓
        # Hidden Layer (128 neurons)
        #     ↓
        # Output Layer (10 neurons)

        self.hidden_weights = [generate_weights(input_size) for _ in range(hidden_size)]
        self.hidden_biases = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                            
        self.output_weights = [generate_weights(hidden_size) for _ in range(output_size)] 
        self.output_biases = [random.uniform(-0.5, 0.5) for _ in range(output_size)]


        self.weights = generate_weights()
        self.bias = random.uniform(-0.5,0.5)
        
        self.learning_rate = 0.01

    def save_weights(self, file: str = "fun.ai"):
        weights_dict = {
            "hidden_weights": self.hidden_weights,
            "hidden_biases": self.hidden_biases,
            "output_weights": self.output_weights,
            "output_biases": self.output_biases
        }
        
        with open(file, 'w') as f:
            json.dump(weights_dict, f)
            
    def load_weights(self, file: str = "fun.ai"):
        with open(file, 'r') as f:
            weights_dict = json.load(f)
            
        self.hidden_weights = weights_dict["hidden_weights"]
        self.hidden_biases = weights_dict["hidden_biases"]
        self.output_weights = weights_dict["output_weights"]
        self.output_biases = weights_dict["output_biases"]
        
    def relu(self, x):
        return max(0, x)
    
        #if input_sum > 0: return input_sum
        #    else: return 0
    

    def softmax(self, out_list):
        max_val = max(out_list)
        exp_x = np.exp([min(x - max_val, 500) for x in out_list])
        exp_sum = np.sum(exp_x)
        epsilon = 1e-10
        return list(exp_x / (exp_sum + epsilon))

    def clip_value(self, x, min_val=-100, max_val=100):
        return max(min(x, max_val), min_val)
            
    def forward(self, inputs_org):
        inputs = inputs_org.copy()

        hidden_outputs = []
        final_outputs = []

        for neuron in range(self.hidden_size):
            res_sum = 0
            for i, weight in enumerate(self.hidden_weights[neuron]):
                res_sum += self.clip_value(inputs[i] * weight)
                
            res_sum += self.hidden_biases[neuron]
            hidden_outputs.append(self.relu(res_sum))

        
        for neuron in range(self.output_size):
            res_sum = 0
            for i, weight in enumerate(self.output_weights[neuron]):
                res_sum += self.clip_value(hidden_outputs[i] * weight)
            
            res_sum += self.output_biases[neuron]
            final_outputs.append(self.relu(res_sum))
        
        return self.softmax(final_outputs), hidden_outputs
    
    
    def train(self, inputs, target):
        outputs, hidden_outputs = self.forward(inputs)
        
        target_array = [0] * self.output_size
        target_array[target] = 1
        
        output_errors = []
        
        for i in range(self.output_size):
            error = target_array[i] - outputs[i]
            output_errors.append(error)
            
                            
        for output_neuron in range(self.output_size):
            error = output_errors[output_neuron]

            for i ,weights in enumerate(self.output_weights[output_neuron]):
                self.output_weights[output_neuron][i] += self.learning_rate * error * hidden_outputs[i]

            self.output_biases[output_neuron] += self.learning_rate * error

        
        for hidden_neuron in range(self.hidden_size):
            error = 0
            for output_neuron in range(self.output_size):
                error += output_errors[output_neuron] * self.output_weights[output_neuron][hidden_neuron]
            
            for i ,weights in enumerate(self.hidden_weights[hidden_neuron]):
                self.hidden_weights[hidden_neuron][i] += self.learning_rate * error * inputs[i]
            
            self.hidden_biases[hidden_neuron] += self.learning_rate * error
                
        
    def predict(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs
        

net = NeuralNet()
    

def train_model():
    
    X, y = load_data('mnist_784.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


    epochs = 10
    best_val_acc = 0
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        correct = 0
        total = 1
        
        val_acc = correct/total
        
        for i in range(len(X_train)):
            net.train(X_train[i], y_train[i])
            
            outputs = net.predict(X_train[i])
            predicted = outputs.index(max(outputs))
            print(predicted, y_train[i])
            if predicted == y_train[i]:
                correct += 1
            total += 1
            
            
            if i % 1000 == 0:
                net.save_weights()
                print(f"Epoch {epoch}, Step {i}, Accuracy: {correct/total:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            net.save_weights('best_model.ai')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break



def ensure_model_trained():
    try:
        net.load_weights()
    except FileNotFoundError:
        print("Training model...")
        train_model()
        net.save_weights()

#train_model()

ensure_model_trained()

X, y = load_data('mnist_784.csv')
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

correct = 0
total = 0
        
for x, y in zip(X_train, y_train):
    outputs = net.predict(x)
    predicted = outputs.index(max(outputs))

    if predicted == y:
        correct += 1
    total += 1
    
    print(f"Predicted digit: {predicted} {y} {correct/total:.2%}")
