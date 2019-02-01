import numpy as np 
import librosa

# This is a static class, but let's use it instead of 
# a bunch of methods because it offers encapsulation.
class Preprocess:
    """Put comments here!
    It will make you happy later when you're
    trying to figure out why you passed in
    a parameter of the wrong datatype and it's
    silently failing.
    """

    directory = "TIMIT/TIMIT/something"

    @staticmethod
    def get_data():
        """Work your magic!
        Should we return the phoneme type as a string or a 
        one-hot encoded vector?

        Returns:
            tuple(list of numpy arrays, list of phoneme type).
        """
        return 2

def test_preprocess():
    result = Preprocess.get_data()
    print("running test_preprocess(); result is {}".format(result))