# SID4SRec
**Sequence- and Item-Level Contrastive Learning via Diffusion-Based Data Augmentation for Sequential Recommendation**

SID4SRec is a sequential recommendation framework that integrates **Diffusion-based Data Augmentation** with **Dual-level Contrastive Learning** (Sequence and Item levels). It enhances item representations by incorporating rich context (Categories and Brands) and leverages the generative power of Diffusion Models to create high-quality augmented samples for self-supervised learning.

---

## Key Features

- **Diffusion-Based Augmentation**: Uses a Gaussian Diffusion process to generate semantically consistent augmented sequences (`aug_seq1`, `aug_seq2`) by denoising latent item representations.
- **Sequence-Level Contrastive Learning**: Implements InfoNCE loss on diffusion-augmented sequences to capture robust user intent.
- **Item-Level Contrastive Learning**: Leverages item metadata (Category/Brand) with an instance-weighting threshold ($\phi$) to filter false negatives and refine item embeddings.
- **Context-Aware Backbone**: A SASRec-style Transformer architecture that fuses Item ID, Category, and Brand embeddings into a unified representation.
- **Distance-based Negative Filtering**: At both the sequence and item levels, preventing semantically similar sequences or items from being incorrectly treated as negatives.

## Environment Setup

- Python 3.8+
- PyTorch 2.0.0+
- Requirements: `pip install -r requirements.txt`

## Quick Start

To train and evaluate the model on a specific dataset (e.g., Beauty):

```bash
python main.py --dataset Beauty --model_name diffsas --gpu_id 0
```

### Key Hyperparameters
- `--alpha`: Weight for Sequence-level CL loss.
- `--lambda_cl`: Weight for Item-level CL loss.
- `--beta`: Weight for Diffusion NLL loss.
- `--phi`: Instance weighting threshold for Item-level CL (filtering false negatives).

## Project Structure

- `main.py`: Entry point for training and evaluation.
- `models/sid4srec.py`: Core implementation of the SID4SRec architecture.
- `models/gaussian_diffusion.py`: Diffusion process and data augmentation logic.
- `trainers/trainer.py`: Training loops and loss computation.
- `data_generators/`: Data preprocessing and context (Category/Brand) injection.
- `configs/`: Dataset-specific hyperparameter configurations.
"# SID4SRec" 
