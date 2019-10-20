#! /usr/bin/env python
"""
train.py

This code uses a neural network built in Keras to classify patients into the following categories:
    normal, mild cognitive impairment (MCI), very mild dementia (VMD) and Severe

The input is an informant-based questionnaire that is taken by someone who knows the patient.
The results of the questionnaire are fed into the network, which uses a primitive grid search for hyperparameters.


"""
#   Imports
###########################################
#   Required Imports
import os, sys
import numpy
import argparse
from utils import debug, get_parser, timeit, parse_arg

#   Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Don't print obnoxious TF warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#   Set the random number seed to ensure repeatable results
numpy.random.seed(7)

#   Debug (print) arguments
###########################################
def debug_args(args, num_classes=4, batch_size=32, hl_neurons=20, hl_activations='tanh',
        dropouts=0.2, output_activation='softmax', epochs=20, learning_rates=0.005, layers=3):
    """Print out all of the arguments if debugging."""
    debug('Number of Classes: %s' % str(num_classes), opt=args)
    debug('Batch Size: %s' % str(batch_size), opt=args)
    debug('Hidden Layer Neurons: %s' % str(hl_neurons), opt=args)
    debug('Hidden Layer Activation Functions: %s' % str(hl_activations), opt=args)
    debug('Dropout Rates: %s' % str(dropouts), opt=args)
    debug('Output Activation Function: %s' % str(output_activation), opt=args)
    debug('Number of Epochs: %s' % str(epochs), opt=args)
    debug('Learning Rates: %s' % str(learning_rates), opt=args)
    debug('Layers: %s' % str(layers), opt=args)
    
    assert 0 < num_classes < 1000, 'Keep number of classes between 1 and 1,000'
    assert 0 < batch_size < 10000, 'Keep batch size between 1 and 10,000'
    assert all(0 < x < 1000 for x in hl_neurons), 'Keep number of hl_neurons between 1 and 1,000'
    assert all(x in ['tanh', 'elu', 'selu', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'] for x in hl_activations), \
        'Activation functions must be one of tanh, elu, selu, sigmoid, hard_sigmoid, exponential, linear'
    assert all(0.0 <= x < 1.0 for x in dropouts), 'Dropout must be >= 0.0 and < 1.0'
    assert all(0 < x < 100000 for x in epochs), 'Keep epochs between 1 and 100,000'
    assert all(0.0 < x < 1.0 for x in dropouts), 'Learning rate must be > 0.0 and < 1.0'
    assert all(0 < x < 1000 for x in layers), 'Keep number of layers between 1 and 1,000'
#   Model
###########################################
@timeit
def compile_and_fit_model(x_train, y_train, x_test, y_test, attributes=37, batch_size=32, epochs=20, hl_neurons=20, 
    dropout=0.2, hl_activation='tanh', output_activation='softmax', learning_rate=0.005, num_classes=4, layers=3):
    """
    Builds, compiles and fits the Keras model.
    
    Params:
        batch_size(int)             How many instances to feed model at a time
        epochs(int)                 How many epochs to train the model for
        hl_neurons(int)             How many neurons per hidden layer
        dropout(float)              0.0-1.0; Percent of neurons to drop in a given hidden layer
        hl_activation(str)          Activation function to use in hidden layer
        
    In CS 611, Artificial Neural Networks, it wsa stated that the optimal number of hl_neurons are:
        (attrs + classifications) / 2
    I have also seen it stated that 2/3 of (attrs + classes) is an optimal number of neurons.
    While there is likely no magic number to work for all ANNs, I have found good results in keeping within this range.
    
    Activation Function types: tanh, elu, selu, sigmoid, hard_sigmoid, exponential, linear
    """
    model = Sequential()
    #   Input layer; does not count toward total number of layers
    model.add(Dense(hl_neurons, activation=hl_activation, input_shape=(attributes,)))

    #   Hidden layers
    for layer in range(layers-1):
        model.add(Dropout(dropout))# avoid overfitting, usually, 0.1,0.2,0.3
        model.add(Dense(hl_neurons, activation=hl_activation))
        
    #   Output layer
    model.add(Dense(num_classes, activation=output_activation))

    model.compile(loss='categorical_crossentropy',  #Loss function，
                optimizer=RMSprop(lr = learning_rate),	#optimization function
                metrics=['accuracy']) # precision, or recall

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,# 1:show you an animated progress, 0 silent, 2 mention the number of epoch, like epoch 1/10 
                        validation_data=(x_test, y_test))
    return model
#   Main (timed with decorator @timeit)
###########################################    
@timeit
def main(args):
    """
    This function is the main body of the program.
    
    First, parse all the command-line arguments, which comprise the hyperparameters.
    Then, debug (print and verify the correct types and values) the hyperparameters.
    Next, load the data.
    Finally, loop through each of the hyperparameter combinations and store the results of each run.
    
    Params:
        args            an argparse.parse_args() object containing the namespace and values of command-line args.
    """
    debug('\n############################################\nSTARTING MAIN FUNCTION IN DEMENTIA DNN\n', opt=args)
    num_classes = args.num_classes
    batch_size = args.batch_size
    layers = parse_arg(args.layers, dtype=int)
    hl_neurons = parse_arg(args.hl_neurons, dtype=int)
    hl_activations = parse_arg(args.hl_activation, dtype=str)
    dropouts = parse_arg(args.dropout, dtype=float)
    output_activation = args.output_activation
    epochs = parse_arg(args.epochs, dtype=int)
    learning_rates = parse_arg(args.learning_rate, dtype=float)

    #   Print out arguments if debugging is on, and ensure that correct values are being used
    debug_args(args, num_classes=num_classes, batch_size=batch_size, hl_neurons=hl_neurons, 
        hl_activations=hl_activations, dropouts=dropouts, output_activation=output_activation, 
        epochs=epochs, learning_rates=learning_rates, layers=layers)

    #
    #   Gather data
    #   Original Headers Removed:
    #   M01	M02	M03	M04	M05	M06	M07	M08	M09	O02	O03	O05	O06	O07	J01	J02	J03	J04	J05	J06	C01	C02	C03	C04	C05	C06	C07	C08	H01	P01	P03	P04	P05	P06	P07	P09	P10	Group
    #   Original Classifications:
    #   0 = Normal, 1 = MCI, 2 = VMD, 3 = Dementia
    #
    trainset = numpy.loadtxt(os.path.join('data', 'train.csv'), delimiter=',')
    testset = numpy.loadtxt(os.path.join('data', 'test.csv'), delimiter=',')
    
    debug('\ntrainset shape: %s, testset shape: %s' % (str(trainset.shape), str(testset.shape)), opt=args)

    attributes = trainset.shape[1] - 1
    debug('Number of attributes: %d' % attributes, opt=args)

    # after comma is column slicing
    x_train = trainset[:,0:attributes]
    y_train = trainset[:,attributes]
    x_test = testset[:,0:attributes]
    y_test = testset[:,attributes]

    debug('x_train shape: %s, y_train shape: %s' % (str(x_train.shape), str(y_train.shape)), opt=args)
    debug('x_test shape: %s, y_test shape: %s' % (str(x_test.shape), str(y_test.shape)), opt=args)

    # convert class vectors to binary representation
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Let's loop!
    # best_score - [accuracy, epochInterval, hl1_activation, hl2_activation, outputActivation, dropout1, dropout2, batch_size, neurons
    best_score = [0.0, epochs[0], hl_activations[0], output_activation, dropouts[0], hl_neurons[0], learning_rates[0]]
    scores = []
    counter = 0
    cumulative_score = 0
    
    for dropout in dropouts:
        for num_hl_neurons in hl_neurons:
            for learning_rate in learning_rates:
                for hl_activation in hl_activations:
                    for num_layers in layers:
                        for num_epochs in epochs:
                            counter += 1
                            print('\n################################\nEXPERIMENT %d' % counter)
                                
                            ###Build, compile and fit the model
                            model = compile_and_fit_model(x_train, 
                                                          y_train, 
                                                          x_test, 
                                                          y_test, 
                                                          attributes=attributes, 
                                                          batch_size=batch_size, 
                                                          epochs=num_epochs, 
                                                          hl_neurons=num_hl_neurons,
                                                          dropout=dropout, 
                                                          hl_activation=hl_activation, 
                                                          output_activation=output_activation, 
                                                          learning_rate=learning_rate,
                                                          num_classes=num_classes,
                                                          layers=num_layers)
                            
                            #   Test metrics
                            test_loss, test_score = model.evaluate(x_test, y_test, verbose=0)
                            
                            #   best_score is a list of [accuracy, <<<hyperparameters>>>]
                            if (test_score > best_score[0]):
                                best_score = [test_score, num_epochs, hl_activation, output_activation, dropout, num_hl_neurons, learning_rate, num_layers]
                            
                            print('num_epochs ' + str(num_epochs) + ', hl_activation ' + str(hl_activation) +  \
                                ', output_activation ' + str(output_activation) + ', dropout ' + str(dropout) + \
                                ', neurons ' + str(num_hl_neurons) + ', Learning Rate: ' + str(learning_rate))
                            print('Test loss: ', test_loss, 'Test accuracy: ', test_score)
                            cumulative_score += test_score
                            
                            #   Adding 
                            scores.append([test_score, num_epochs, hl_activation, output_activation, dropout, num_hl_neurons, learning_rate, num_layers])
                        #   END for num_epochs in epochs    
                    # END for num_layers in layers
                #   END for hl_activation in hl_activations 
            #   END for learning_rate in learning_rates
        #   END for num_hl_neurons in hl_neurons
    #   for dropout in dropouts
    
    print('\nBest accuracy: \n%s\n%s' % ('[test_score, num_epochs, hl_activation, output_activation, dropout, num_hl_neurons, learning_rate, num_layers]', str(best_score)))
    sorted_average = sorted(scores, key=lambda x: (x[1]))
    for score in sorted_average:
        print (str(score))
    print('Average accuracy: %f' % (cumulative_score / counter))

#   Main
###########################################    
if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--layers',default=3,help='Number of layers; to search multiple layer options, separate integers by comma (2,3,4)')
    parser.add_argument('--dropout',default=0.2,help='Dropout; to search multiple dropouts, separate floats by comma (0.2,0.3,0.4)')
    parser.add_argument('--hl_neurons',default=20,help='Number of neurons in hidden layer; to search multiple options, separate integers by comma (20,30,40)')
    parser.add_argument('--learning_rate',default=0.005,help='Learning rate; to search multiple LRs, separate floats by comma (0.2,0.3,0.4)')
    parser.add_argument('--hl_activation',default='tanh',help='Hidden layer activation; to search multiple activation options, separate integers by comma (tanh,sigmoid)')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size; must be integer.')
    parser.add_argument('--epochs',default=20,help='Number of epochs to train model; to search multiple options, separate integers by comma (20,30,40)')
    parser.add_argument('--num_classes',type=int,default=4,help='Number of classifications for our model.')
    parser.add_argument('--output_activation',type=str,default='softmax',help='Output activation function.')
    args = parser.parse_args()
	
    #   Run timed main section with args
    main(args)