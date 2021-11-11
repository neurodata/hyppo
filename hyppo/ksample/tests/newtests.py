import abc
import numpy
from hyppo.ksample import SmoothCFTest, MeanEmbeddingTest
from unittest import TestCase
from numpy.random import seed


class StatTests:
    @abc.abstractmethod
    def validate(self,pvalue):
        pass

    def test_SmoothCF(self):
        pvalue = SmoothCFTest(self.X, self.Y, scale=2.0**(-5)).compute_pvalue()
        self.validate(pvalue)

    def test_analytic(self):
        pvalue = MeanEmbeddingTest(self.X,self.Y).compute_pvalue()
        self.validate(pvalue)

class TestNull(TestCase,StatTests):
    def setUp(self):
        seed(120)
        num_samples = 500
        dimensions = 10
        self.X = numpy.random.randn(num_samples, dimensions)
        self.Y = numpy.random.randn(num_samples, dimensions)

    def validate(self,pvalue):
        self.assertGreater(pvalue, 0.05)


class TestAlternative(TestCase,StatTests):
    def setUp(self):
        seed(120)
        num_samples = 500
        dimensions = 10
        self.X = numpy.random.randn(num_samples, dimensions)
        self.X[:, 1] *= 3
        self.Y = numpy.random.randn(num_samples, dimensions)

    def validate(self,pvalue):
        self.assertLess(pvalue, 0.05)

