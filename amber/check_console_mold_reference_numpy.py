import numpy as np

from src.algorithm.util.console_mold_reference import _solution_to_element_energy_density
# 4-6

def main():
    vertex_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    element_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)
    solution = vertex_positions[:, 0]

    importance = _solution_to_element_energy_density(
        element_indices=element_indices,
        vertex_positions=vertex_positions,
        solution=solution,
    )
    assert importance.shape == (1,)
    assert np.isclose(float(importance[0]), 1.0, atol=1.0e-6)
    print("Console/Mold reference energy-density check passed.")


if __name__ == "__main__":
    main()
