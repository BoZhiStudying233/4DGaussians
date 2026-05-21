_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
)
OptimizationParams=dict(
    coarse_iterations = 8000,

    densify_grad_threshold_coarse = 0.00001,#0.0002
    densify_grad_threshold_fine_init =0.000005,
    prune_threshold = 40000,
    grow_threshold = 1500_000,
    opacity_threshold_coarse = 0.6,#0.005
    opacity_threshold_fine_init = 0.2,#0.005
    opacity_threshold_fine_after = 0.4,#0.005
    
    coarse_densify_threshold = 1500_000,
    fine_densify_threshold = 200_000,
    attn_scale = 100,

)