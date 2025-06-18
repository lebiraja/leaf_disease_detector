import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

class ModelConfig:
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    DATA_DIR = "dataset"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    MODEL_OUTPUT_DIR = "models"
    MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_model.pth")
    CLASS_NAMES_PATH = os.path.join(MODEL_OUTPUT_DIR, "class_names.txt")
    IMAGE_SIZE = (224, 224)
    NUM_EPOCHS = 10  # Optimized for ~1.5 hours
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda")
    FP16 = True

def load_data_from_dirs(config: ModelConfig):
    logger.info("--- Starting Data Loading ---")
    train_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_test_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        for dir_path in [config.TRAIN_DIR, config.VALID_DIR]:
            if not os.path.exists(dir_path):
                logger.error(f"Directory not found: {dir_path}")
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            if not os.listdir(dir_path):
                logger.error(f"Directory is empty: {dir_path}")
                raise ValueError(f"Directory is empty: {dir_path}")

        train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(config.VALID_DIR, transform=valid_test_transforms)
        test_dataset = None
        if os.path.exists(config.TEST_DIR) and os.listdir(config.TEST_DIR):
            test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=valid_test_transforms)
            logger.info("Test dataset loaded")
        else:
            logger.warning("Test dataset not found or empty; skipping test evaluation")

        if not train_dataset.classes:
            logger.error("No classes found in training dataset")
            raise ValueError("No classes found in training dataset")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

    class_counts = np.bincount(train_dataset.targets)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = class_weights_sampler[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0)
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0)
        )

    logger.info(f"Loaded {len(train_dataset)} training images, {len(valid_dataset)} validation images")
    if test_dataset:
        logger.info(f"Loaded {len(test_dataset)} test images")
    logger.info(f"Classes: {train_dataset.classes}")
    return train_loader, valid_loader, test_loader, train_dataset.classes, train_dataset.targets

class ImageClassifierTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, class_weights, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.class_weights = class_weights
        self.config = config
        self.device = config.DEVICE
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scaler = torch.cuda.amp.GradScaler() if config.FP16 else None
        self.logger = logging.getLogger(__name__)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            try:
                if self.config.FP16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            except RuntimeError as e:
                self.logger.error(f"Training error: {e}")
                raise
            batch_loss = loss.item() * inputs.size(0)
            running_loss += batch_loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=batch_loss / inputs.size(0))
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        progress_bar.close()
        return epoch_loss, epoch_acc

    def evaluate(self, loader, desc="Evaluating", class_names=None):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(len(class_names))} if class_names else {}
        class_total = {i: 0 for i in range(len(class_names))} if class_names else {}
        progress_bar = tqdm(loader, desc=desc, leave=False)
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                try:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                except RuntimeError as e:
                    self.logger.error(f"Evaluation error: {e}")
                    raise
                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if class_names:
                    for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                        class_total[label] += 1
                        class_correct[label] += (label == pred)
                progress_bar.set_postfix(loss=batch_loss / inputs.size(0))
        loss = running_loss / total
        acc = 100 * correct / total
        if class_names:
            for i, cls in enumerate(class_names):
                class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                self.logger.info(f"{cls}: {class_acc:.2f}%")
        progress_bar.close()
        return loss, acc

    def train(self):
        self.logger.info("--- Starting Model Training on GPU ---")
        best_valid_loss = float('inf')
        os.makedirs(self.config.MODEL_OUTPUT_DIR, exist_ok=True)
        for epoch in range(self.config.NUM_EPOCHS):
            try:
                train_loss, train_acc = self.train_epoch()
                valid_loss, valid_acc = self.evaluate(self.valid_loader, desc="Validating", class_names=self.train_loader.dataset.classes)
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                    f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%"
                )
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                    self.logger.info(f"Saved best model to {self.config.MODEL_SAVE_PATH}")
            except Exception as e:
                self.logger.error(f"Epoch {epoch+1} failed: {e}")
                raise
        if self.test_loader:
            test_loss, test_acc = self.evaluate(self.test_loader, desc="Testing", class_names=self.train_loader.dataset.classes)
            self.logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        else:
            self.logger.info("No test dataset; skipping test evaluation")
        self.logger.info("--- Training Complete ---")

def train_model():
    logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    cfg = ModelConfig()
    try:
        train_loader, valid_loader, test_loader, class_names, train_targets = load_data_from_dirs(cfg)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    try:
        with open(cfg.CLASS_NAMES_PATH, 'w') as f:
            f.write(str(class_names))
        logger.info(f"Class names saved to {cfg.CLASS_NAMES_PATH}")
    except Exception as e:
        logger.error(f"Failed to save class names: {e}")
        raise
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_targets),
        y=train_targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Loading model: {cfg.MODEL_NAME}")
    try:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if cfg.PRETRAINED else None)
        num_classes = len(class_names)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(cfg.DEVICE)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainer = ImageClassifierTrainer(model, train_loader, valid_loader, test_loader, class_weights, cfg)
    trainer.train()

if __name__ == '__main__':
    try:
        torch.cuda.empty_cache()
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise