# Sparse Autoencoders Trained on the Same Data Learn Different Features
Acompanying code for our research on SAE feature overlap when trained on different seeds.

We provide a script to calculate the alignment between two SAEs using the hungarian algorithm, [hungarian_alignment.py](hungarian_alignment.py).

We also provide a script to plot all the figures in the paper, [sae_overlap.ipynb](sae_overlap.ipynb).

Although the aligment of the SAEs with less than 32k latents is fast, the aligment of the other sizes takes a long time, so we provide the average aligment and the indices in the [alignment](alignment) folder.


