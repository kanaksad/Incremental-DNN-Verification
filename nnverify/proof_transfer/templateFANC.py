import copy
import torch

class TemplateStoreFANC:
    def __init__(self):
        # Map templates by layer and output constraint, with lists for multiple templates
        self.template_map = {}

    def get_template(self, input, output_constraint):
        """
        Retrieve all template details for a specific layer and output constraint.
        """
        layer_map = self.template_map.get(input, None)
        if layer_map is not None:
            return layer_map.get(output_constraint, [])
        return []

    def add_template_detail(self, layer, lb, ub, output_constraint, input):
        """
        Add a template detail indexed by layer and output constraint.
        """
        if tuple(input.tolist()) not in self.template_map:
            self.template_map[tuple(input.tolist())] = {}
        if output_constraint not in self.template_map[tuple(input.tolist())]:
            self.template_map[tuple(input.tolist())][output_constraint] = []
        
        # Append new template details to the list
        self.template_map[tuple(input.tolist())][output_constraint].append({
            "layer": layer,
            "lb": copy.deepcopy(lb),
            "ub": copy.deepcopy(ub),
        })

    def contains2(self, layer, lb, ub, output_constraint, threshold=1):
        """
        Check if a given lb and ub are contained within any template for the specified
        layer and output constraint, based on a percentage threshold.

        Parameters:
        - layer: Layer number.
        - lb: Lower bound tensor.
        - ub: Upper bound tensor.
        - output_constraint: Output constraint associated with the template.
        - threshold: Minimum percentage of elements that must be within bounds to consider containment (default is 0.9).

        Returns:
        - bool: True if the percentage of elements within bounds exceeds the threshold, otherwise False.
        """
        templates = self.get_template(layer, output_constraint)

        # Iterate over each template to check containment
        for template in templates:
            template_lb = template["lb"]
            template_ub = template["ub"]

            # Calculate the element-wise containment ratios
            lb_within_ratio = (lb >= template_lb).float().mean().item()
            ub_within_ratio = (ub <= template_ub).float().mean().item()

            # Check if both lb and ub meet or exceed the threshold
            if lb_within_ratio >= threshold and ub_within_ratio >= threshold:
                return True  # Contained within this template based on the threshold

        return False  # No template satisfied the containment condition

    
    def contains(self, layer, lb, ub, output_constraint, input):
        """
        Check if a given lb and ub are contained within any template for the specified
        layer and output constraint.
        """
        templates = self.get_template(tuple(input.tolist()), output_constraint)

        # Iterate over each template to check containment
        for template in templates:
            if template["layer"] == layer:
                template_lb = template["lb"]
                template_ub = template["ub"]

                # Check if lb and ub are contained within template_lb and template_ub
                lb_within = torch.all(lb >= template_lb)
                ub_within = torch.all(ub <= template_ub)

                if lb_within and ub_within:
                    return True  # Contained within this template

        return False  # Not contained within any template
    def print_map_size(self):
        """
        Print the total number of templates stored in the map and the distribution by layer.
        """
        total_count = 0
        for input, constraint_map in self.template_map.items():
            layer_count = sum(len(templates) for templates in constraint_map.values())
            total_count += layer_count
            # print(f"Layer {layer} has {layer_count} templates.")

        print(f"Total templates stored: {total_count}")