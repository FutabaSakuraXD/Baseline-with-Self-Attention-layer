# Baseline-with-Self-Attention-layer
Self-Attn layer is an implementation of SAGAN(Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena.Self-Attention Generative Adversarial Networks.ICML 2019)
This model is composed of a baseline(MGN) with Self-attention applied behind specific layers.
Since element-wise multiplication is used in self attention layer, this model might take tons of CUDA memory. 
