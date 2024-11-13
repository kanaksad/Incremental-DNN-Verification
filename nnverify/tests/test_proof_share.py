from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.bnb import Split
from nnverify.analyzer import Analyzer
import nnverify.config as config
import nnverify.inc_proof_share.proof_share_main as ps
import ssl

# This is a hack to avoid SSL related errors while getting CIFAR10 data
from nnverify.specs.input_spec import InputSpecType
from nnverify.smoothing.code.predict import SmoothingAnalyzer, SmoothingArgs

        
class TestMNISTIncremental(TestCase):
    def test_mnist_base_diffai(self):
        args = ps.ShareArgs(net=config.MNIST_PGD_6_500_3, tuned_net=config.MNIST_PGD_6_500_3, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6])
        ps.proof_share(args)
        # args = config.Args(net=config.MNIST_PGD_6_500_3, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0003, k = 3)
        # Analyzer(args).run_analyzer()
        
    def test_mnist_base_diffai_first_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_FIRST_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.001, k = 3)
        Analyzer(args).run_analyzer()
    def test_mnist_base_diffai_second_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_SECOND_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.001)
        Analyzer(args).run_analyzer()
    def test_mnist_base_diffai_third_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_THIRD_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.001)
        Analyzer(args).run_analyzer()
    def test_mnist_base_diffai_forth_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_FORTH_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.001)
        Analyzer(args).run_analyzer()
    def test_mnist_base_diffai_fifth_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_FIFTH_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_diffai_all_block(self):
        args = config.Args(net=config.MNIST_DIFFAI_5_100_ALL_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.001)
        Analyzer(args).run_analyzer()
    
    
    def test_mnist_base_pgd_1(self):
        args = config.Args(net=config.MNIST_PGD_6_500_1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    
    def test_mnist_base_pgd_3(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_first_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_FIRST_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_second_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_SECOND_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_third_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_THIRD_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_forth_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_FORTH_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_fifth_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_FIFTH_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_sixth_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_SIXTH_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    def test_mnist_base_pgd_3_all_block(self):
        args = config.Args(net=config.MNIST_PGD_6_500_3_ALL_BLOCK, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.015)
        Analyzer(args).run_analyzer()
    