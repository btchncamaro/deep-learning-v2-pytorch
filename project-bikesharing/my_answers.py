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
        # TODO: Hidden layer - Replace these values with your calculations.
        #hidden_inputs = None # signals into hidden layer
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)
        #hidden_outputs = None # signals from hidden layer
        hidden_outputs = np.array([[self.activation_function(i) for i in hidden_inputs]])  # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        #final_inputs = None # signals into final output layer
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        #final_outputs = None # signals from final output layer
        final_outputs = np.sum(final_inputs)
        
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
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        #error = None # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs  # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        #hidden_error = None
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #output_error_term = None
        output_error_term = error * final_outputs * (1 - final_outputs)

        print("output_error_term: {}, weights_hidden_to_output: {}".format(output_error_term, self.weights_hidden_to_output))
        print("output_error_term.shape: {}, weights_hidden_to_output.shape: {}".format(output_error_term.shape, self.weights_hidden_to_output.shape))
        #hidden_error = np.dot(output_error_term, self.weights_hidden_to_output)
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)

        #hidden_error_term = None
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        #delta_weights_i_h += None
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        print("\noutput_error_term: {}, hidden_outputs.T: {}".format(output_error_term, hidden_outputs.T))
        print("output_error_term.shape: {}, hidden_outputs.T.shape: {}".format(output_error_term.shape, hidden_outputs.T.shape))
        #delta_weights_h_o += None
        #delta_weights_h_o += output_error_term * hidden_outputs
        #delta_weights_h_o += np.dot(output_error_term, hidden_outputs) #ValueError: non-broadcastable output operand with shape (2,1) doesn't match the broadcast shape (2,2)
        dot_result = np.dot(hidden_outputs.T, output_error_term)
        print("delta_weights_h_o.shape: {}, delta_weights_h_o: {}".format(delta_weights_h_o.shape, delta_weights_h_o))
        delta_weights_h_o += dot_result.T
        #delta_weights_h_o += hidden_outputs * output_error_term

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #self.weights_hidden_to_output += None # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += delta_weights_h_o/n_records  # update hidden-to-output weights with gradient descent step
        #self.weights_input_to_hidden += None # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h/n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        #hidden_inputs = None # signals into hidden layer
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden)
        #hidden_outputs = None # signals from hidden layer
        hidden_outputs = np.array([self.activation_function(i) for i in hidden_inputs])
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        #final_inputs = None # signals into final output layer
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        #final_outputs = None # signals from final output layer
        final_outputs = np.sum(final_inputs)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
