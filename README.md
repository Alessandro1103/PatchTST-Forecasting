# Alessandro De Luca: PatchTST-Forecasting

In this GitHub repository, we present the study and the corresponding implemented code of the paper titled "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers".

PatchTST is a Transformer-based framework designed for multivariate time series forecasting that treats time series as a sequence of "patches" (sub-sequences) rather than individual time steps. (However, due to computational constraints, our implementation utilizes a scaled-down backbone with d_model=128 instead of the standard 512, optimized for CPU execution, while maintaining the standard depth of 3 layers). The purpose is to extract local semantic information from time series while significantly reducing the computational complexity of the Attention mechanism, enabling the model to learn long-term dependencies effectively.

The model is based on Patching, a process that segments the input series into overlapping tokens, and Channel Independence, where each variate is embedded and processed independently by the Transformer backbone. To handle the non-stationary nature of time series data, we integrated RevIN (Reversible Instance Normalization), which normalizes the input to zero mean and unit variance before the backbone and denormalizes the output, mitigating the distribution shift problem.

In the Self-Supervised Pre-training phase, we employed a random masking strategy (controlled by mask_ratio). Inspired by the success of Masked Language Modeling in NLP (e.g., BERT) and Masked Autoencoders in Vision, the model learns to reconstruct missing patches, acquiring robust representations of the data without labels. For Supervised Forecasting, the pre-trained backbone is fine-tuned with a linear head (head_forecast) that projects the latent representations to the prediction horizon (e.g., 96 future steps).

We validated the approach on the ETT (Electricity Transformer Temperature) dataset, comparing a model trained from scratch versus a fine-tuned model. All the specific implementations and the comparative analysis are described in the notebook.

[Google Colab Notebook](https://colab.research.google.com/drive/11GJyRLQ5xPXld0j9332I0OIzc1Nj_Fy_#scrollTo=WbuslqS_wZpe)
