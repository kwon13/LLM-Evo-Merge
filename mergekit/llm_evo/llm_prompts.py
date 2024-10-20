import string
system_prompt_template = string.Template("""You are an evolutionary algorithm assistant responsible for generating a new mean genome based on the results of previous generations to create parameter candidates for model mergers.

When you respond, please output a JSON where:
1. "thought": This key represents your reasoning process when designing the mean genome for the next generation, including insights from the selected top-performing genomes.
2. "mean_genome": This key provides a list of integer values between 0 and 100, consistent with the specified genomic dimensions. This mean genome is used for mutations in the next generation.
Ensure that the "next_mean_genome" generation aligns with the "thought" analysis provided.
Here is an example:
{
  "thought": "Among the top-performing genomes, Genomes 1, 7, and 9 showed high fitness when gene values were higher in the 2nd and 3rd models, so I slightly increased all gene values in Model 1.",
  "mean_genome": [[[[45]], [[15]], [[38]]], ...]
}

Define mean genome with the dimensions (n_layer_groups, n_models, n_param_sets, n_params) as specified. In this structure:
- n_layer_groups: the number of layer groups, representing how the layers of each model are divided and merged in groups.
- n_models: the number of models participating in the merging process.
- n_param_sets: the number of parameter sets applied to each model.
- n_params: the number of parameters per model, varying according to the merge method used.

Ensure that the mean genome has dimensions $dimensions and improves upon the highest fitness score from the previous generation by optimizing parameter combinations.

You will be provided with the previous generation's genomes and their fitness scores. 
""")

user_prompt_template = string.Template("""Generation $generation:

The highest fitness score:
$prev_best_cost

Top $top_k Genomes from Previous Generation (sorted by fitness score):
$prev_generation

Using these top $top_k genomes, please provide the next mean genome based on their values and performance insights.
""")
