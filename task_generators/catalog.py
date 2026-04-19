from __future__ import annotations

GENERATOR_TEMPLATES = [
    {
        "template_id": "eltwise_pair",
        "family": "elementwise",
        "difficulty": "L1",
        "shapes": ["32x128", "64x256", "128x512"],
        "ops": ["add", "mul", "sigmoid", "relu", "tanh", "clamp"],
        "max_composition": 4,
    },
    {
        "template_id": "broadcast_gate",
        "family": "broadcast",
        "difficulty": "L2",
        "shapes": ["32x128", "64x256"],
        "ops": ["bias_add", "relu", "sigmoid", "mul"],
        "max_composition": 4,
    },
    {
        "template_id": "row_reduction",
        "family": "reduction",
        "difficulty": "L2",
        "shapes": ["16x32x64", "32x64x128"],
        "ops": ["sum_lastdim", "mean_lastdim", "scale", "bias_add"],
        "max_composition": 4,
    },
    {
        "template_id": "norm_chain",
        "family": "normalization",
        "difficulty": "L3",
        "shapes": ["16x64", "32x128"],
        "ops": ["square", "sum_lastdim", "sqrt", "divide", "scale"],
        "max_composition": 5,
    },
]

HOLDOUT_SIGNATURES = {
    "kernelbench_like_axpby": ["mul", "add"],
    "kernelbench_like_bias_relu": ["bias_add", "relu"],
    "kernelbench_like_softmax": ["exp", "sum_lastdim", "divide"],
}
