import torch
import matplotlib.pyplot as plt

repaird_lbs_ubs_path = 'results/nnverify-nets-aprnn-repaired-mnist_relu_9_100_polytope.pthDomain.DEEPPOLY.pt'
unrepaird_lbs_ubs_path = 'results/nnverify-nets-aprnn-unrepaired-mnist_relu_9_100.pthDomain.DEEPPOLY.pt'

repaird_lbs_ubs = torch.load(repaird_lbs_ubs_path)
unrepaird_lbs_ubs = torch.load(unrepaird_lbs_ubs_path)

unrepaired_lbs = unrepaird_lbs_ubs['lbs']
unrepaired_ubs = unrepaird_lbs_ubs['ubs']

repaired_lbs = repaird_lbs_ubs['lbs']
repaired_ubs = repaird_lbs_ubs['ubs']

for i, (unrepaired_lb, unrepaired_ub, repaired_lb, repaired_ub) in enumerate(zip(unrepaired_lbs, unrepaired_ubs, repaired_lbs, repaired_ubs)):
    print(f"\nBounds for proof {i+1}:")
    
    if unrepaired_lb.numel() == 0 or unrepaired_ub.numel() == 0 or repaired_lb.numel() == 0 or repaired_ub.numel() == 0:
        print("One or more tensors are empty, skipping comparison.")
        continue

    if unrepaired_lb.shape == unrepaired_ub.shape == repaired_lb.shape == repaired_ub.shape:
        for j in range(unrepaired_lb.numel()): 
            lb_unrepaired = unrepaired_lb.view(-1)[j].item() 
            ub_unrepaired = unrepaired_ub.view(-1)[j].item()
            lb_repaired = repaired_lb.view(-1)[j].item()
            ub_repaired = repaired_ub.view(-1)[j].item()

            print(f"[{lb_unrepaired}, {ub_unrepaired}], [{lb_repaired}, {ub_repaired}]")

        unrepaired_lb_flat = unrepaired_lb.detach().view(-1).numpy()
        unrepaired_ub_flat = unrepaired_ub.detach().view(-1).numpy()
        repaired_lb_flat = repaired_lb.detach().view(-1).numpy()
        repaired_ub_flat = repaired_ub.detach().view(-1).numpy()

        plt.figure(figsize=(10, 6))
        elements = range(len(unrepaired_lb_flat))

        plt.plot(elements, unrepaired_lb_flat, 'r--', label='Unrepaired LB')
        plt.plot(elements, unrepaired_ub_flat, 'r-', label='Unrepaired UB')

        plt.plot(elements, repaired_lb_flat, 'g--', label='Repaired LB')
        plt.plot(elements, repaired_ub_flat, 'g-', label='Repaired UB')

        plt.xlabel('Element Index')
        plt.ylabel('Bound Value')
        plt.title(f'Comparison of Unrepaired vs Repaired Bounds for Proof {i+1}')
        plt.legend()

        plt.show()

    else:
        print(f"Tensors in set {i+1} do not have the same shape, skipping comparison.")
