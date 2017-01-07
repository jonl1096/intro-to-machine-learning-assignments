import math

from ml_utils import Hw5Utils

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self._n = p.chain_length()
        self._k = p.num_x_values()

        self._p = [] # p before normalization

    def marginal_probability(self, x_i):
        """ Calculates the marginal probability for the ith node.

        Args:
            x_i: The ith node's i value.

        Returns:
            A python list of type float, with its length=k+1, and the first value 0

        """

        message_left = [0.0] * (self._k + 1)

        message_right = [0.0] * (self._k + 1)

        # print('left')
        for i in range(1, x_i):
            # print(i)
            if i == 1:
                message_left = self.calculate_unary_message(i)
            else:
                message_left = Hw5Utils.element_wise_multiply(self.calculate_unary_message(i), message_left)
            # ^ this is u_x -> f

            message_left = self.calculate_mu_to_x_message_left(self._n + i, message_left)
            # ^ this is u_f -> x

        # print('message_left:')
        # print(message_left)

        # print('right')
        for i in range(self._n, x_i, -1):
            # print(i)
            if i == self._n:
                message_right = self.calculate_unary_message(i)
            else:
                message_right = Hw5Utils.element_wise_multiply(self.calculate_unary_message(i), message_right)

            # print(self._n + i - 1)
            message_right = self.calculate_mu_to_x_message_right(self._n + i - 1, message_right)

        # print('message_right')
        # print(message_right)

        p = []
        if message_left == [0] * len(message_left):
            p = message_right
        elif message_right == [0] * len(message_left):
            p = message_left
        else:
            p = Hw5Utils.element_wise_multiply(message_left, message_right)

        # print('message_right times message_left')
        # print(p)
        m = self.calculate_unary_message(x_i)

        # print('message_up')
        # print(m)

        p = Hw5Utils.element_wise_multiply(m, p)
        self._p = p

        # print('p')
        # print(p)

        if sum(p) != 1:
            p = Hw5Utils.normalize(p)

        return p

        # # This code is used for testing only and should be removed in your implementation.
        # # It creates a uniform distribution, leaving the first position 0
        # result = [1.0 / (self._potentials.num_x_values())] * (self._potentials.num_x_values() + 1)
        # result[0] = 0
        # return result

    def calculate_mu_to_x_message_left(self, node_index, previous_message):
        message = [0] * (self._k + 1) # x values begin at 1
        for i in range(1, self._k + 1):
            for j in range(1, self._k + 1):
                message[i] += self._potentials.potential(node_index, j, i) * previous_message[j]

        return message

    def calculate_mu_to_x_message_right(self, node_index, previous_message):
        message = [0] * (self._k + 1) # x values begin at 1
        for i in range(1, self._k + 1):
            for j in range(1, self._k + 1):
                message[i] += self._potentials.potential(node_index, i, j) * previous_message[j]

        return message

    def calculate_unary_message(self, node_index):
        message = [0] * (self._k + 1)
        for i in range(1, self._k + 1):
            message[i] = self._potentials.potential(node_index, i)
        return message

    def get_normalization_constant(self):
        return Hw5Utils.get_normalization_constant(self._p)



class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        self._n = p.chain_length()
        self._k = p.num_x_values()
        self._messages_k = [0] * (p.chain_length() + 1)

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):

        # print('x_i = %d' % x_i)
        message_left = [0.0] * (self._k + 1)

        message_right = [0.0] * (self._k + 1)

        # print('left')
        for i in range(1, x_i):
            # print(i)
            if i == 1:
                message_left = self.calculate_unary_message(i)
            else:
                message_left = Hw5Utils.element_wise_add(self.calculate_unary_message(i), message_left)

            # self._assignments[i] = self.get_max_k(message_left)
            # self._assignments[i] = self.get_max_k(self.calculate_unary_message(i))
            # print('m.l. before')
            # print(message_left)
            message_left, message_k_left = self.calculate_mu_to_x_message_left(self._n + i, message_left)
            # print('m.l. after')
            # print(message_left)
            self._messages_k[i] = message_k_left

        # print('right')
        for i in range(self._n, x_i, -1):
            # print(i)
            if i == self._n:
                message_right = self.calculate_unary_message(i)
            else:
                message_right = Hw5Utils.element_wise_add(self.calculate_unary_message(i), message_right)

            # self._assignments[i] = self.get_max_k(message_right)
            # self._assignments[i] = self.get_max_k(self.calculate_unary_message(i))

            # print(self._n + i - 1)
            # print('m.r. before')
            # print(message_right)
            message_right, message_k_right = self.calculate_mu_to_x_message_right(self._n + i - 1, message_right)
            # print('m.r. after')
            # print(message_right)
            self._messages_k[i] = message_k_right

        # print('messages')
        # print(self._messages)
        #
        # print('message_left:')
        # print(message_left)
        #
        # print('message_right')
        # print(message_right)

        p = []

        # print('message_right times message_left')
        # print(p)
        m = self.calculate_unary_message(x_i)

        # print('message_up')
        # print(m)

        p = [x + y for x, y in zip(message_left, message_right)]

        p = [x + y for x, y in zip(p, m)]

        # print('p')
        # print(p)

        self._assignments[x_i] = self.get_max_k(p)

        p = self.normalize(p, x_i)

        max_prob = self.get_max_prob(p)
        max_k = self.get_max_k(p)
        self.calculate_assignments(x_i, max_k)

        return max_prob

    def calculate_assignments(self, x_i, k):
        max_k = int(k)
        for i in range(x_i - 1, 0):
            message_k = self._messages_k[i]
            self._assignments[i] = message_k[max_k]
            max_k = int(message_k[max_k])

        max_k = int(k)
        for i in range(x_i + 1, self._n + 1):
            message_k = self._messages_k[i]
            # print('i: %d' % i)
            # print('max_k: %d' % max_k)
            self._assignments[i] = message_k[max_k]
            max_k = int(message_k[max_k])


    def calculate_mu_to_x_message_left(self, node_index, previous_message):
        message = [0] * (self._k + 1) # x values begin at 1
        message_k = [0] * (self._k + 1) # x values begin at 1
        for i in range(1, self._k + 1):
            # max_p = 0.0
            max_p = float('-inf')
            max_k = 0
            for j in range(1, self._k + 1):
                p = math.log(self._potentials.potential(node_index, j, i)) + previous_message[j]
                if p > max_p:
                    max_p = p
                    max_k = j
            message[i] = max_p
            message_k[i] = max_k

        return message, message_k


    def calculate_mu_to_x_message_right(self, node_index, previous_message):
        message = [0] * (self._k + 1) # x values begin at 1
        message_k = [0] * (self._k + 1) # x values begin at 1
        for i in range(1, self._k + 1):
            # max_p = 0.0
            max_p = float('-inf')
            max_k = 0
            for j in range(1, self._k + 1):
                p = math.log(self._potentials.potential(node_index, i, j)) + previous_message[j]
                if p > max_p:
                    max_p = p
                    max_k = j
            message[i] = max_p
            message_k[i] = max_k

        return message, message_k

    def calculate_unary_message(self, node_index):
        message = [0] * (self._k + 1)
        for i in range(1, self._k + 1):
            message[i] = math.log(self._potentials.potential(node_index, i))
        return message

    def get_max_k(self, message):
        # max_p = 0
        max_p = float('-inf')
        max_k = 0
        for j in range(1, self._k + 1):
            if message[j] > max_p:
                max_p = message[j]
                max_k = j

        return max_k

    def get_max_prob(self, message):
        # max_p = 0
        max_p = float('-inf')
        max_k = 0
        for j in range(1, self._k + 1):
            if message[j] > max_p:
                max_p = message[j]
                max_k = j

        return max_p

    def normalize(self, vector, x_i):
        sp = SumProduct(self._potentials)
        sp.marginal_probability(x_i)
        z = sp.get_normalization_constant()
        # print('z: %f' % z)

        result = [x - math.log(z) for x in vector]

        return result

