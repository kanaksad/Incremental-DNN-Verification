import copy
import torch

class TemplateStoreFANC:
    def __init__(self):
        # Map templates by layer and output constraint, with lists for multiple templates
        self.template_map = {}

    def get_template(self, layer, output_constraint):
        """
        Retrieve all template details for a specific layer and output constraint.
        """
        layer_map = self.template_map.get(layer, None)
        if layer_map is not None:
            return layer_map.get(output_constraint, [])
        return []

    def add_template_detail(self, layer, lb, ub, output_constraint, input):
        """
        Add a template detail indexed by layer and output constraint.
        """
        if layer not in self.template_map:
            self.template_map[layer] = {}
        if output_constraint not in self.template_map[layer]:
            self.template_map[layer][output_constraint] = []
        
        # Append new template details to the list
        self.template_map[layer][output_constraint].append({
            "layer": layer,
            "lb": copy.deepcopy(lb),
            "ub": copy.deepcopy(ub),
            "output_constraint": output_constraint,
            "input": input
        })

    def get_all_constraints_for_layer(self, layer):
        """
        Retrieve all templates for a given layer.
        """
        return self.template_map.get(layer, {})
    
    def contains(self, layer, lb, ub, output_constraint):
        """
        Check if a given lb and ub are contained within any template for the specified
        layer and output constraint.
        """
        templates = self.get_template(layer, output_constraint)

        # Iterate over each template to check containment
        for template in templates:
            template_lb = template["lb"]
            template_ub = template["ub"]

            # Check if lb and ub are contained within template_lb and template_ub
            lb_within = torch.all(lb >= template_lb)
            ub_within = torch.all(ub <= template_ub)

            if lb_within and ub_within:
                return True  # Contained within this template

        return False  # Not contained within any template