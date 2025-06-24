import subprocess
import itertools

# Definition of possible values
k_reals_list = [10, 30, 50]
synt_neigh_size_list = [200, 600, 1000]

# Definition of perturbation types: only one at a time
perturbations = ["ont_perturbation", "gen_perturbation"]

# Loop over all parameters
for perturbation in perturbations:
    for k_reals, synt_neigh_size in itertools.product(k_reals_list, synt_neigh_size_list):
        command = [
            "python",
            "jaccard_neigh.py",
            "--k_reals", str(k_reals),
            "--synt_neigh_size", str(synt_neigh_size),
            f"--{perturbation}",
        ]
        print("Running command:", " ".join(command))
        subprocess.run(command)