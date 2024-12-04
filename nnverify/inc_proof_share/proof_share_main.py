from nnverify import config
from nnverify.analyzer import Analyzer
from nnverify.common.result import Result, Results
from nnverify.proof_transfer.pt_util import compute_speedup, plot_verification_results
from nnverify.proof_transfer.pt_types import ProofTransferMethod
from nnverify.specs.input_spec import InputSpecType


class ShareArgs:
    def __init__(self, domain, k=0, tuned_net='', pt_method=None, count=None, spec_type=InputSpecType.LINF, eps=0.01, dataset='mnist', attack=None,
                 split=None, net='',
                 timeout=30):
        self.net = config.NET_HOME + net
        self.domain = domain
        self.pt_method = pt_method
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.attack = attack
        self.split = split
        self.tuned_net = config.NET_HOME + tuned_net
        self.timeout = timeout
        self.spec_type = spec_type
        self.k = k

    def set_net(self, net):
        self.net = config.NET_HOME + net

    def get_verification_arg(self, net):
        arg = config.Args(net=self.net, k = self.k, domain=self.domain, dataset=self.dataset, eps=self.eps,
                          split=self.split, count=self.count, pt_method=self.pt_method, timeout=self.timeout)
        # net is set correctly again since the home dir is added here
        if net == None:
            arg.net = self.net
        else:
            arg.net = self.tuned_net
        return arg


def proof_share(pt_args):
    # print("proof share initiated")
    res, res_pt = proof_transfer_analyze(pt_args)
    # speedup = compute_speedup(res, res_pt, pt_args)
    # print("Proof Transfer Speedup :", speedup)
    # plot_verification_results(res, res_pt, pt_args)
    # return speedup


# def proof_share_analyze(pt_args):
#     args = pt_args.get_verification_arg()
#     analyzer = Analyzer(args)
#     _ = analyzer.run_analyzer()
#     template_store = analyzer.template_store
#     if args.pt_method == ProofTransferMethod.ALL:
#         # precomputes reordered template store
#         template_store = get_reordered_template_store(args, template_store)
#     approx_net = get_perturbed_network(pt_args)
#     # Use template generated from original verification for faster verification of the approximate network
#     approx_args = pt_args.get_verification_arg()
#     res_pt = Analyzer(approx_args, net=approx_net, template_store=template_store).run_analyzer()
#     # Compute results without any template store as the baseline
#     res = Analyzer(args, net=approx_net).run_analyzer()
#     return res, res_pt


def proof_transfer_acas(pt_args):
    res = Results(pt_args)
    res_pt = Results(pt_args)
    for i in range(1, 6):
        for j in range(1, 10):
            pt_args.set_net(config.ACASXU(i, j))
            pt_args.count = 4
            r, rp = proof_transfer_analyze(pt_args)
            res.results_list += r.results_list
            res_pt.results_list += rp.results_list

    # compute merged stats
    res.compute_stats()
    res_pt.compute_stats()

    speedup = compute_speedup(res, res_pt, pt_args)
    print("Proof Transfer Speedup :", speedup)
    plot_verification_results(res, res_pt, pt_args)
    return speedup


def proof_transfer_analyze(pt_args):
    args = pt_args.get_verification_arg(net=None)

    res_fst = Analyzer(args, reuse=True).run_analyzer()
    analyzer = Analyzer(args)
    # run analyzer until layer k
    # create templates that can verify the properties
    #KD:make sure that we have a decent template store at first
    res_org = analyzer.run_analyzer()
    template_store_fanc = analyzer.template_store
    # if args.pt_method == ProofTransferMethod.ALL:
    #     # precomputes reordered template store
    #     template_store = get_reordered_template_store(args, template_store)
    # approx_net = get_perturbed_network(pt_args)
    # # Use template generated from original verification for faster verification of the approximate network
    tuned_args = pt_args.get_verification_arg(net=pt_args.tuned_net)
    # res_pt = Analyzer(approx_args, net=approx_net, template_store=template_store).run_analyzer()
    # Compute results without any template store as the baseline
    #KD: use template store once established
    res_tuned = Analyzer(tuned_args, template_store=template_store_fanc, reuse=True).run_analyzer()
    return res_org, res_tuned


def get_reordered_template_store(args, template_store):
    args.pt_method = ProofTransferMethod.REORDERING
    # Compute reordered template
    analyzer_reordering = Analyzer(args, template_store=template_store)
    # TODO: make this as a separate a function from run_analyzer that takes some budget for computing
    _ = analyzer_reordering.run_analyzer()
    # This template store should contain leaf nodes of reordered tree
    template_store = analyzer_reordering.template_store
    return template_store


def get_perturbed_network(pt_args):
    # Generate the approximate network
    approx_net = pt_args.tuned_net
    return approx_net
