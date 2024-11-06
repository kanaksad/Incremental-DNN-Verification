import torch
import copy

from torch.nn import ReLU, Linear, Conv2d
from onnx import numpy_helper
from nnverify.common.network import Layer, LayerType, Network


def forward_layers(net, relu_mask, transformers):
    #KD: add the logic here to see if we forward from here, is the property still verifiable
    # list all the layers first
    templates = []
    lsize = len(net)
    for i in range(lsize):
        if net[i].type == LayerType.ReLU:
            transformers.handle_relu(net[i], optimize=True, relu_mask=relu_mask)
            # map_relu_layer_idx_to_layer_idx_temp = copy.deepcopy(transformers.map_relu_layer_idx_to_layer_idx)
            # map_for_noise_indices_temp = copy.deepcopy(transformers.map_for_noise_indices)
            # TODO: deepcopy can be expensive. should do it lightly. work on the commmented code.
            if i <= (2*lsize / 3):
                find_template(net, relu_mask, copy.deepcopy(transformers), i + 1, lsize, templates)
            
            # transformers.cofs = transformers.cofs[:(i + 1)]
            # transformers.centers = transformers.centers[:(i + 1)]
            # transformers.relu_layer_cofs = transformers.relu_layer_cofs[:(i + 1)]
            # transformers.unstable_relus = transformers.unstable_relus[:(i + 1)]
            # transformers.map_relu_layer_idx_to_layer_idx = map_relu_layer_idx_to_layer_idx_temp
            # transformers.map_for_noise_indices = map_for_noise_indices_temp
        elif net[i].type == LayerType.Linear:
            if net[i] == net[-1]:
                transformers.handle_linear(net[i], last_layer=True)
            else:
                transformers.handle_linear(net[i])
        elif net[i].type == LayerType.Conv2D:
            transformers.handle_conv2d(net[i])
        elif net[i].type == LayerType.Normalization:
            transformers.handle_normalization(net[i])
    transformers.templates = templates
    return transformers

"""
create a template.
"""
def find_template(net, relu_mask, transformers, starting_layer, network_layer_count, templates=None):
    # get a template
    eps_cur = 1
    while(eps_cur > 0.01):
        copied_transformer = copy.deepcopy(transformers)
        eps_cur /= 2
        adjusted_lbs = copied_transformer.lbs[-1] - eps_cur
        adjusted_ubs = copied_transformer.ubs[-1] + eps_cur
        copied_transformer.ubs[-1] = adjusted_ubs
        copied_transformer.lbs[-1] = adjusted_lbs
        # see if the template can verify stuffs
        forward_layers_with_template(net, relu_mask, copied_transformer, starting_layer, network_layer_count, templates)
        lb = copied_transformer.compute_lb()
        if torch.all(lb >= 0):
            # print("VERIFIED Template")
            template_detail = {
                "layer": starting_layer,
                "lb": copy.deepcopy(adjusted_lbs),
                "ub": copy.deepcopy(adjusted_ubs),
                "output_constraint": transformers.prop.out_constr
            }
            templates.append(template_detail)
            break
        # else: 
            # print("UNKNOWN Template")

"""
create a copy of the transformer. forward everything from the layer told. 
"""
def forward_layers_with_template(net, relu_mask, transformers, starting_layer, network_layer_count, templates=None):
    # get a template
    # see if the template can verify stuffs
    for current_layer in range(starting_layer, network_layer_count):
        if net[current_layer].type == LayerType.ReLU:
            transformers.handle_relu(net[current_layer], optimize=True, relu_mask=relu_mask)
        elif net[current_layer].type == LayerType.Linear:
            if net[current_layer] == net[-1]:
                transformers.handle_linear(net[current_layer], last_layer=True)
            else:
                transformers.handle_linear(net[current_layer])
        elif net[current_layer].type == LayerType.Conv2D:
            transformers.handle_conv2d(net[current_layer])
        elif net[current_layer].type == LayerType.Normalization:
            transformers.handle_normalization(net[current_layer])
    # lb = transformers.compute_lb()
    # if torch.all(lb >= 0):
    #     print("VERIFIED")
    # else: 
    #     print("UNKNOWN")

# def forward_layers(net, relu_mask, transformers):
#     #KD: add the logic here to see if we forward from here, is the property still verifiable
#     # list all the layers first
    
#     for layer in net:
#         if layer.type == LayerType.ReLU:
#             transformers.handle_relu(layer, optimize=True, relu_mask=relu_mask)
#         elif layer.type == LayerType.Linear:
#             if layer == net[-1]:
#                 transformers.handle_linear(layer, last_layer=True)
#             else:
#                 transformers.handle_linear(layer)
#         elif layer.type == LayerType.Conv2D:
#             transformers.handle_conv2d(layer)
#         elif layer.type == LayerType.Normalization:
#             transformers.handle_normalization(layer)
#     return transformers


"""
check if the template holds.
"""

"""
store the template if it
"""

def parse_onnx_layers(net):
    input_shape = [dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim]

    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape, net_format='onnx')
    num_layers = len(net.graph.node)
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in
                              net.graph.initializer}

    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input

        if operation == 'MatMul':
            # Assuming that the add node is followed by the MatMul node
            add_node = net.graph.node[cur_layer + 1]
            bias = model_name_to_val_dict[add_node.input[1]]

            # Making some weird assumption that the weight is always 0th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], bias=bias, type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Conv':
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Conv2D)
            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            layers.append(layer)

        elif operation == 'Gemm':
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Relu':
            layers.append(Layer(type=LayerType.ReLU))

    return layers


def parse_torch_layers(net):
    layers = Network(torch_net=net, net_format='torch')

    for torch_layer in net:
        if isinstance(torch_layer, ReLU):
            layers.append(Layer(type=LayerType.ReLU))
        elif isinstance(torch_layer, Linear):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear)
            layers.append(layer)
        elif isinstance(torch_layer, Conv2d):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias,
                          type=LayerType.Conv2D)
            layer.kernel_size = torch_layer.kernel_size
            layer.padding = torch_layer.padding
            layer.stride = torch_layer.stride
            layer.dilation = (1, 1)
            layers.append(layer)

    return layers