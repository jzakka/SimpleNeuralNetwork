import numpy as np

LEARNING_RATE = 0.3
    
# forward propagation 함수는 레이어를 다음 레이어의 입력을 만들어냅니다    
def forward_propagate(input_data, weight):
    product = np.dot(input_data, weight)
    return sigmoid(product)

# backward propagation 함수는 이전 레이어로부터의 가중치를 조절합니다
def backward_propagate(weight, inputs, error):
    copied_weight = np.copy(weight)
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            error_delta = get_error_delta(copied_weight, i, j, inputs, error[0, j])
            weight[i, j] -= LEARNING_RATE * error_delta

# 이전 레이어로 에러를 배분합니다
def distribute_error(weight, error):
    prev_error = np.zeros((1, error.shape[1]))
    for j in range(error.shape[1]):
        weight_sum = np.sum(weight[:, j])
        
        for i in range(weight.shape[0]):
            prev_error[0, i] += weight[i, j] / weight_sum * error[0, j]
    return prev_error

def sigmoid(input): 
    return np.exp(input) / (1.0 + np.exp(input))

def get_sum_of_sigmoid(weight, inputs, j):
    return sigmoid(np.sum(weight[:, j] * inputs[0, :]))

def get_error_delta(weight, i, j, inputs, error):
    sum_of_sigmoid = get_sum_of_sigmoid(weight, inputs, j)
    return -error * sum_of_sigmoid * (1 - sum_of_sigmoid) * inputs[0, i]

def get_error(target, actual):
    return target - actual

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

class ThreeByThreeNeuralNet:
    def __init__(self, weights1, weights2):
        self.weights1 = weights1
        self.weights2 = weights2

    # 3x3 뉴럴네트워크에서 학습을 진행합니다
    # 1. forward propagation을 진행하고
    # 2. 각 레이어로 에러를 배분합니다
    # 3. 마지막으로 backward propagation을 진행하여 가중치를 조절합니다.
    def learn(self, input_data, target, log_on):
        first_output = forward_propagate(input_data, self.weights1)
        final_output = forward_propagate(first_output, self.weights2)

        layer3_error = get_error(target, final_output)
        layer2_error = distribute_error(self.weights2, layer3_error)

        backward_propagate(self.weights2, first_output, layer3_error)
        backward_propagate(self.weights1, input_data, layer2_error)
        
        if log_on:
            print("Output of layer2 is")
            print_matrix(first_output)
            print()
            print("Output of layer3 is")
            print_matrix(final_output)
            print()
            print("Error of layer3 is")
            print_matrix(layer3_error)
            print()
            print("Error of layer2 is")
            print_matrix(layer2_error)
            print()
            print("Weights from layer2 to layer 3 is")
            print_matrix(self.weights2)
            print()
            print("Weights from layer1 to layer 2 is")
            print_matrix(self.weights1)
            print()

    def process(self, input_data):
        return forward_propagate(forward_propagate(input_data, self.weights1), self.weights2)        

neural_net = ThreeByThreeNeuralNet(
    np.array([
        [0.9, 0.2, 0.1],
        [0.3, 0.8, 0.5],
        [0.4, 0.2, 0.6]
    ]),
    np.array([
        [0.3, 0.6, 0.8],
        [0.7, 0.5, 0.1],
        [0.5, 0.2, 0.9]
    ])
)

# 교재 예시와 같은 입력
input_data = np.array([[0.9, 0.1, 0.8]])

# 학습 목표를 설정 : [0.6 0.5 0.2]
target = np.array([[0.6, 0.5, 0.2]])

non_learned_output = neural_net.process(input_data)

print("Output from neuralNet which not learned is")
print_matrix(non_learned_output)
print()

# 로그를 통해 학습의 진행과정을 확인 가능합니다.
neural_net.learn(input_data, target, True)

# 총 1000번의 학습을 진행
for _ in range(1000):
    neural_net.learn(input_data, target, False)

result_after_thousands_learned = neural_net.process(input_data)

print("Result after learned 1000 times is")
print_matrix(result_after_thousands_learned)