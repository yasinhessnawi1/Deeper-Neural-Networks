# Generative AI

Assignment 6 for the Deeper Neural Networks course. Covers denoising autoencoders, VAEs, GANs, and diffusion models on Fashion-MNIST.

## Contents

| File | Description |
|------|-------------|
| `part_a_denoising_ae.py` | Denoising autoencoder: reconstructs images corrupted with Gaussian noise or label overlay |
| `part_b_vae.py` | Variational autoencoder: ELBO loss, reparameterization trick, sample generation from latent space |
| `part_c_gan.py` | DCGAN-style GAN: generator and discriminator, adversarial training |
| `part_d_diffusion.py` | DDPM: forward diffusion, U-Net denoiser, reverse sampling |
| `compare_all.py` | Side-by-side comparison of samples from all four models. Requires Parts A–D to be run first. |
| `report.tex` | LaTeX report for the assignment |

## Running

From the project root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
python generative-ai/part_a_denoising_ae.py
python generative-ai/part_b_vae.py
python generative-ai/part_c_gan.py
python generative-ai/part_d_diffusion.py
python generative-ai/compare_all.py
```

Each part saves figures and sample tensors to the `generative-ai/` folder. Run `compare_all.py` last to produce the combined comparison figure.
