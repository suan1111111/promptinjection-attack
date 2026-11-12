# RCS kcd.py commit 3ea8e29f3542e7962a7eaed7996dc5654b4035fd
# Definitions below copied unchanged; standalone Qwen configuration.
import torch
import torch.nn as nn
import torch.nn.functional as F
class ProjectionConfig:
    """Global configuration for projection training methods"""

    # Projection training mode
    # "single_layer": Train one projection on layer 16, use for all layers (original method)
    # "layer_specific": Train separate projection for each layer (new method)
    PROJECTION_MODE: str = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER: int = 16  # Which layer to use for training projection

    # Training hyperparameters
    PROJECTION_EPOCHS: int = 100
    PROJECTION_BATCH_SIZE: int = 64
    PROJECTION_LEARNING_RATE: float = 1e-3
    PROJECTION_LR_SCHEDULER: str = "cosine"
    PROJECTION_STEP_SIZE: int = 25
    PROJECTION_GAMMA: float = 0.5
    PROJECTION_COSINE_MIN_LR_FACTOR: float = 0.05
    PROJECTION_MAX_PATIENCE: int = 15

    # Architecture settings (will be set dynamically based on model)
    INPUT_DIM: int = 4096  # Default for LLaVA/InternVL, will be updated for Qwen (2048)
    OUTPUT_DIM: int = 256
    HIDDEN_DIM: int = 512
    DROPOUT: float = 0.3

    # Loss weighting settings
    DATASET_LOSS_WEIGHT: float = 1.0  # Weight for dataset classification loss
    TOXICITY_LOSS_WEIGHT: float = (
        5.0  # Weight for toxicity detection loss (higher to balance magnitude)
    )

    @classmethod
    def set_model_dimensions(cls, model_type, input_dim=None):
        """Set input dimensions based on model type and actual feature dimensions"""
        if model_type.lower() == "qwen":
            # Use actual input dimension if provided, otherwise use default
            cls.INPUT_DIM = input_dim if input_dim is not None else 2048
            cls.OUTPUT_DIM = 256  # Reduce output dimension for Qwen
            cls.HIDDEN_DIM = 512  # Reduce hidden dimension for Qwen
            cls.PROJECTION_EPOCHS = 100  # Reduce epochs for Qwen
        else:
            # Use actual input dimension if provided, otherwise use default
            cls.INPUT_DIM = input_dim if input_dim is not None else 4096

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("PROJECTION CONFIGURATION")
        print("=" * 80)
        print(f"Mode: {cls.PROJECTION_MODE}")
        if cls.PROJECTION_MODE == "single_layer":
            print(f"Training layer: {cls.SINGLE_LAYER_TRAINING_LAYER}")
        print(f"Architecture: {cls.INPUT_DIM} -> {cls.HIDDEN_DIM} -> {cls.OUTPUT_DIM}")
        print(
            f"Training: {cls.PROJECTION_EPOCHS} epochs, batch_size={cls.PROJECTION_BATCH_SIZE}"
        )
        print(
            f"Learning rate: {cls.PROJECTION_LEARNING_RATE} (scheduler={cls.PROJECTION_LR_SCHEDULER}, "
            f"cosine_factor={cls.PROJECTION_COSINE_MIN_LR_FACTOR}, step_size={cls.PROJECTION_STEP_SIZE}, gamma={cls.PROJECTION_GAMMA})"
        )
        print(
            f"Loss weights: Dataset={cls.DATASET_LOSS_WEIGHT}, Toxicity={cls.TOXICITY_LOSS_WEIGHT}"
        )
        print("=" * 80)

CONFIG = ProjectionConfig()
CONFIG.set_model_dimensions("qwen")

class LearnedProjection(nn.Module):
    """
    Learned projection from 4096 to 256 dimensions with multi-objective loss:
    1. Samples from same dataset should be close
    2. Samples from different datasets should be far
    3. Benign centroids should be close, malicious centroids should be close,
       but benign and malicious centroids should be far apart
    """

    def __init__(self, input_dim=None, output_dim=None, hidden_dim=None, dropout=None):
        super(LearnedProjection, self).__init__()

        # Use config defaults if not specified
        input_dim = input_dim or CONFIG.INPUT_DIM
        output_dim = output_dim or CONFIG.OUTPUT_DIM
        hidden_dim = hidden_dim or CONFIG.HIDDEN_DIM
        dropout = dropout or CONFIG.DROPOUT

        # Multi-layer projection network
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through projection network"""
        return self.projection(x)

def compute_contrastive_loss(
    embeddings, dataset_labels, toxicity_labels, margin_dataset=1.0, margin_toxicity=2.0
):
    """
    Compute multi-objective contrastive loss:
    1. Dataset clustering: same dataset samples close, different dataset samples far
    2. Toxicity separation: benign centroids close, malicious centroids close,
       but benign and malicious far apart
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # Normalize embeddings for better training stability
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 1. Dataset clustering loss
    dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)
    unique_datasets = torch.unique(dataset_labels)

    if len(unique_datasets) > 1:
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for same/different datasets
        dataset_same_mask = (
            dataset_labels.unsqueeze(0) == dataset_labels.unsqueeze(1)
        ).float()
        dataset_diff_mask = 1.0 - dataset_same_mask

        # Remove diagonal (self-comparisons)
        eye_mask = 1.0 - torch.eye(batch_size, device=device)
        dataset_same_mask *= eye_mask
        dataset_diff_mask *= eye_mask

        # Dataset clustering loss: minimize intra-dataset distances, maximize inter-dataset distances
        if dataset_same_mask.sum() > 0:
            intra_dataset_loss = (
                distances * dataset_same_mask
            ).sum() / dataset_same_mask.sum()
        else:
            intra_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if dataset_diff_mask.sum() > 0:
            inter_dataset_loss = (
                torch.clamp(margin_dataset - distances, min=0) * dataset_diff_mask
            )
            inter_dataset_loss = inter_dataset_loss.sum() / dataset_diff_mask.sum()
        else:
            inter_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        dataset_loss = intra_dataset_loss + inter_dataset_loss
    else:
        dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Toxicity separation loss
    toxicity_loss = torch.tensor(0.0, device=device, requires_grad=True)
    unique_toxicity = torch.unique(toxicity_labels)

    if len(unique_toxicity) > 1:
        # Compute centroids for each toxicity class
        benign_mask = toxicity_labels == 0
        malicious_mask = toxicity_labels == 1

        if benign_mask.sum() > 0 and malicious_mask.sum() > 0:
            benign_embeddings = embeddings[benign_mask]
            malicious_embeddings = embeddings[malicious_mask]

            benign_centroid = benign_embeddings.mean(dim=0)
            malicious_centroid = malicious_embeddings.mean(dim=0)

            # Distance between benign and malicious centroids (should be large)
            centroid_distance = F.pairwise_distance(
                benign_centroid.unsqueeze(0), malicious_centroid.unsqueeze(0)
            )

            # Loss: maximize distance between centroids
            toxicity_loss = torch.clamp(
                margin_toxicity - centroid_distance, min=0
            ).mean()

            # Add intra-class compactness
            if len(benign_embeddings) > 1:
                benign_distances = torch.norm(
                    benign_embeddings - benign_centroid.unsqueeze(0), dim=1
                )
                toxicity_loss = toxicity_loss + benign_distances.mean()

            if len(malicious_embeddings) > 1:
                malicious_distances = torch.norm(
                    malicious_embeddings - malicious_centroid.unsqueeze(0), dim=1
                )
                toxicity_loss = toxicity_loss + malicious_distances.mean()

    # Combine losses with configurable weights
    weighted_dataset_loss = CONFIG.DATASET_LOSS_WEIGHT * dataset_loss
    weighted_toxicity_loss = CONFIG.TOXICITY_LOSS_WEIGHT * toxicity_loss
    total_loss = weighted_dataset_loss + weighted_toxicity_loss

    return total_loss, dataset_loss, toxicity_loss
