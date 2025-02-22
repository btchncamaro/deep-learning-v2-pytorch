import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))


    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # DONE: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)

        #hidden_outputs = None # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # DONE: Output layer - Replace these values with your calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs   #do not use activation function for final output.  Do not use derivative for this during backprob.
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # print("Starting backpropagation: ")
        # print("    final_outputs:     {}".format(final_outputs))
        # print("    hidden_outputs:    {}".format(hidden_outputs))
        # print("    X (input):         {}".format(X))
        # print("    y (labels):        {}".format(y))
        # print("    delta_weights_i_h: {}".format(delta_weights_i_h))
        # print("    delta_weights_h_o: {}".format(delta_weights_h_o))

        #### Implement the backward pass here ####
        ### Backward pass ###

        # DONE: Output error - Replace this value with your calculations.
        error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

        # DONE: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error #not using activation function for error, so no derivative needed here.

        # DONE: Calculate the hidden layer's contribution to the error

        # Weight step (hidden to output)
        delta_weights_h_o += (output_error_term * hidden_outputs.T)[:, None]

        hidden_error = np.matmul(output_error_term, self.weights_hidden_to_output.T)  # hidden error shape: (2, 1), hidden_outputs shape: (1, 2)
        hidden_error_term = hidden_error.T * hidden_outputs.T * (1 - hidden_outputs.T)

        # Weight step (input to hidden)
        delta_weights_i_h += np.matmul(hidden_error_term[:, None], X[:, None].T).T

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''

        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # DONE: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = np.array([self.activation_function(i) for i in hidden_inputs])
        
        # DONE: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  #np.sum(final_inputs)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# iterations = 5000
# learning_rate = 0.15
# hidden_nodes = 25
# output_nodes = 1
#Progress: 100.0% ... Training loss: 0.221 ... Validation loss: 0.387

# iterations = 5000
# learning_rate = 0.2
# hidden_nodes = 25
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.180 ... Validation loss: 0.321

# iterations = 8000
# learning_rate = 0.2
# hidden_nodes = 25
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.085 ... Validation loss: 0.197

iterations = 8000
learning_rate = 0.25
hidden_nodes = 25
output_nodes = 1
#Progress: 100.0% ... Training loss: 0.072 ... Validation loss: 0.168