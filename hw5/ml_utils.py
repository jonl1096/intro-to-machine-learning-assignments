
class Hw5Utils:

    @staticmethod
    def normalize(vector):
        z = sum(vector)
        vector = [x / float(z) for x in vector]
        return vector

    @staticmethod
    def get_normalization_constant(vector):
        return sum(vector)

    @staticmethod
    def element_wise_multiply(vect1, vect2):
        return [a * b for a, b in zip(vect1, vect2)]

    @staticmethod
    def element_wise_add(vect1, vect2):
        return [x + y for x, y in zip(vect1, vect2)]
