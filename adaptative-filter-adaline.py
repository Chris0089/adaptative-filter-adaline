import random
import pylab
import numpy
import matplotlib.pyplot as plt

N = 3
FILENAME = 'signal.txt'
OUTPUT_FILE = 'output.txt'
ETA = 0.0004
DWN_LIM_RAND_VAL = -1
UP_LIM_RAND_VAL = 1
ACTIVATION_FUNCTION_DERIV = 1


class DataAccessObject:
    desiredOutput = []

    def __init__(self):
        self.clean_output_file()
        self.get_data()

    def clean_output_file(self):
        open(OUTPUT_FILE, 'w').close()

    def write_output_value(self, output):
        file = open(OUTPUT_FILE, 'a')
        file.write(str(output))
        file.write(" ")

    def get_data(self):
        file = open(FILENAME, "r")
        for line in file:
            fields = line.split("\t")
            size = len(fields)
            for position in range(0, size):
                self.desiredOutput.append(float(fields[position]))

class Adaline(DataAccessObject):
    inputValues = []
    weightValues = []
    outputValues = []
    desiredOutput = []
    bias = 0.0
    error = 0.0
    y = 0.0
    min_val_y = 0
    max_val_y = 0

    def __init__(self):
        self.inputValues = DataAccessObject.desiredOutput.copy()
        self.desiredOutput = DataAccessObject.desiredOutput.copy()
        self.generate_random_weights()
        self.bias = random.uniform(DWN_LIM_RAND_VAL, UP_LIM_RAND_VAL)
        self.adaptative_filter()
        self.plot_outputs()

    def generate_random_weights(self):
        for quantity in range(0,N):
            self.weightValues.append(random.uniform(DWN_LIM_RAND_VAL, UP_LIM_RAND_VAL))

    def output_initial_values(self):
        for quantity in range(0,N):
            self.outputValues.append(self.inputValues[quantity])

    def calculate_output(self, inputList, weightList):
        summation = 0
        for position in range(0,len(inputList)):
            summation = summation + (inputList[position] * weightList[position])
        threshold = summation + self.bias
        return self.activation_function(threshold)

    def activation_function(self, threshold):
        return threshold

    def adaptative_filter(self):
        self.output_initial_values()
        size_output = len(self.inputValues)
        for i in range(0, size_output-N):
            print ("iteracion " + str(i))
            self.y = 0.0
            self.error = 0.0
            inputList = [ ]
            for q in range(0, N-2):
                inputList.append(self.inputValues[i+q])
            self.y = self.calculate_output(inputList, self.weightValues)
            self.error = self.desiredOutput[i+N] - self.y
            self.training(i)
            self.outputValues.append(self.y)
        self.write_output_file()

    def training(self, inputValuePosition):
        for i in range(0, len(self.weightValues)):
            self.weightValues[i] = self.weightValues[i] + \
                                   (ETA * self.error * ACTIVATION_FUNCTION_DERIV *
                                    self.inputValues[inputValuePosition+i])
        self.bias = self.bias + (ETA * self.error * ACTIVATION_FUNCTION_DERIV)


    def write_output_file(self):
        for outputValue in self.outputValues:
            DataAccessObject.write_output_value(self, outputValue)

    def get_limits_y(self):
        self.min_val_y = min(self.desiredOutput)
        self.max_val_y = max(self.desiredOutput)

    def plot_outputs(self):
        self.get_limits_y()
        x = numpy.linspace(0, len(self.desiredOutput), len(self.desiredOutput))
        pylab.plot(x, self.desiredOutput, )
        plt.axis([0, len(self.desiredOutput), self.min_val_y-1, self.max_val_y+1])
        pylab.show()

        x2 = numpy.linspace(0, len(self.outputValues), len(self.outputValues))
        pylab.plot(x2, self.outputValues, 'm')
        plt.axis([0, len(self.outputValues), self.min_val_y-1, self.max_val_y+1])
        pylab.show()

        pylab.plot(x, self.desiredOutput, )
        pylab.plot(x2, self.outputValues, 'm')
        plt.axis([0, len(self.outputValues), self.min_val_y-1, self.max_val_y+1])
        pylab.show()



dao = DataAccessObject()
adaptative_filter = Adaline()
print(adaptative_filter.outputValues)
print(adaptative_filter.desiredOutput)
