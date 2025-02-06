# sia_music_gen

This is something I developed out of curisity for generating symbolic music in midi format. I have tried several different models such as LSTM, LSTM with attention, transformer with chord-aware attention, Vector Quantized Variational Auto Encoder (VQ-VAE), & Diffusion Transformer.


python 3.12.8, Torch 2.15.1 with CUDA 12.4

The basic models to run LSTM with attention is in chord_aware_LSTM.py

For Vector-Quantized VAE run:

```bash
python sia_vqvae_diff.py
```

Then once the model is created run latent diffusion by:

```
python sia_vqvae_diffTrans_seq.py
```
