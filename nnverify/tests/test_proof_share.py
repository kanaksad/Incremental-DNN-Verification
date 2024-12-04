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
    def test_with_exp_losses(self):
        import torch
        model = torch.load('nnverify/nets/sota/mnist-1.pth', map_location="cuda:0")
        print("hi")
        
        # args = ps.ShareArgs(net=config.MNIST_CNN_7_1_CCIBP, tuned_net=config.MNIST_CNN_7_1_CCIBP, domain=Domain.BOX, dataset=Dataset.MNIST, eps=.1, k = [6], count=100)
        # res_fst = Analyzer(args, reuse=True).run_analyzer()

    # def test_mnist_base_diffai(self):
    #     args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=1000)
    #     ps.proof_share(args)
        # args = config.Args(net=config.MNIST_PGD_6_500_3, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0003, k = 3)
        # Analyzer(args).run_analyzer()
    
    def test1(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=.1, k = [6], count=100)
        res_fst = Analyzer(args, reuse=True).run_analyzer()
    
    def test2(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)
        
    def test3(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        # if args.pt_method == ProofTransferMethod.ALL:
        #     # precomputes reordered template store
        #     template_store = get_reordered_template_store(args, template_store)
        # approx_net = get_perturbed_network(pt_args)
        # # Use template generated from original verification for faster verification of the approximate network
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        # res_pt = Analyzer(approx_args, net=approx_net, template_store=template_store).run_analyzer()
        # Compute results without any template store as the baseline
        #KD: use template store once established
        import pickle
        with open("template_store1.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        res_tuned = Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()       


    def test1cache(self):
        cache_val = 0.1
        import pickle
        with open("template_store1.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()
        
    def test03cache(self):
        cache_val = 0.03
        import pickle
        with open("template_store03.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()
                         
    def test01cache(self):
        cache_val = 0.01
        import pickle
        with open("template_store01.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()       
        
    def test005cache(self):
        cache_val = 0.005
        import pickle
        with open("template_store005.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()
        
    def test0025cache(self):
        cache_val = 0.0025
        import pickle
        with open("template_store0025.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
    def test001cache(self):
        cache_val = 0.001
        import pickle
        with open("template_store001.pkl", "rb") as file:
            template_store_fanc = pickle.load(file)
        
        print("verifying 0.3 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.1 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.03 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.01 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.005 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer() 
        
        print("verifying 0.0025 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()  
        
        print("verifying 0.001 with cache ", cache_val)
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        tuned_args = args.get_verification_arg(net=args.tuned_net)
        Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()
        

    def collect3(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.3, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store3.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)  
            
    def collect1(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.1, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store1.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)  

    def collect03(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.03, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store03.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)  

    def collect01(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.01, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store01.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)    

    def collect005(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.005, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store005.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)
            
    def collect0025(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0025, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store0025.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)
            
            
    def collect001(self):
        args = ps.ShareArgs(net=config.MNIST_DIFFAI_5_100, tuned_net=config.MNIST_DIFFAI_5_100, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.001, k = [6], count=10000)
        analyzer = Analyzer(args)
        # run analyzer until layer k
        # create templates that can verify the properties
        #KD:make sure that we have a decent template store at first
        res_org = analyzer.run_analyzer()

        import pickle
        with open("template_store001.pkl", "wb") as file:
            pickle.dump(analyzer.template_store, file)

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