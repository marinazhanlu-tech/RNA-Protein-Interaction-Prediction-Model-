import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def setup_logging(log_dir, log_file='training.log'):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for batch_idx, (rna_inputs, protein_inputs, targets) in enumerate(progress_bar):
        rna_inputs = rna_inputs.to(device)
        protein_inputs = protein_inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(rna_inputs, protein_inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        if writer and batch_idx % 100 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train', loss.item(), step)
    
    avg_loss = running_loss / len(train_loader)
    logging.info(f'Train Epoch {epoch+1}: Average Loss: {avg_loss:.6f}')
    return avg_loss

def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
        for rna_inputs, protein_inputs, targets in progress_bar:
            rna_inputs = rna_inputs.to(device)
            protein_inputs = protein_inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(rna_inputs, protein_inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({'val_loss': loss.item(), 'acc': 100.*correct/total})
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    logging.info(f'Validation Epoch {epoch+1}: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    if writer:
        writer.add_scalar('Loss/Val', avg_loss, epoch)
        writer.add_scalar('Accuracy/Val', accuracy, epoch)
    
    return avg_loss, accuracy

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)

def main():
    # Import project modules
    from models import RnaProteinInteractionModel
    from data_loader import RNADataset, create_data_loader
    from config import get_config
    from utils import generate_dummy_data
    
    # Configuration
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    log_dir = config['paths']['log_path']
    checkpoint_dir = config['paths']['model_save_path']
    
    # Setup logging
    setup_logging(log_dir)
    logging.info(f'Training on device: {device}')
    logging.info(f'Model config: {config["model"]}')
    logging.info(f'Training config: {config["training"]}')
    
    # Initialize model
    model = RnaProteinInteractionModel(**config['model']).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    # 使用更温和的学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    
    # Load training data from CSV file (excluding test and val sets)
    data_file = os.path.join('data', 'rna_protein_data.csv')
    val_file = os.path.join('data', 'rna_protein_val.csv')
    test_file = os.path.join('data', 'rna_protein_test.csv')
    
    if os.path.exists(data_file):
        from utils import load_data_from_csv
        logging.info(f'Loading training data from {data_file}')
        all_rna, all_protein, all_labels = load_data_from_csv(data_file)
        
        # Load validation and test sets to exclude them from training data
        val_rna, val_protein, val_labels = [], [], []
        test_rna, test_protein, test_labels = [], [], []
        
        if os.path.exists(val_file):
            logging.info(f'Loading validation data from {val_file}')
            val_rna, val_protein, val_labels = load_data_from_csv(val_file)
        
        if os.path.exists(test_file):
            logging.info(f'Loading test data from {test_file}')
            test_rna, test_protein, test_labels = load_data_from_csv(test_file)
        
        # Exclude validation and test sets from training data
        if val_rna and test_rna:
            val_set = set(zip(val_rna, val_protein))
            test_set = set(zip(test_rna, test_protein))
            train_rna, train_protein, train_labels = [], [], []
            for r, p, l in zip(all_rna, all_protein, all_labels):
                if (r, p) not in val_set and (r, p) not in test_set:
                    train_rna.append(r)
                    train_protein.append(p)
                    train_labels.append(l)
            logging.info(f'Excluded validation ({len(val_labels)}) and test ({len(test_labels)}) sets from training data')
        elif val_rna:
            # Only validation set exists
            val_set = set(zip(val_rna, val_protein))
            train_rna, train_protein, train_labels = [], [], []
            for r, p, l in zip(all_rna, all_protein, all_labels):
                if (r, p) not in val_set:
                    train_rna.append(r)
                    train_protein.append(p)
                    train_labels.append(l)
            logging.info(f'Excluded validation set ({len(val_labels)}) from training data')
        else:
            # No separate validation/test sets, use all data for training
            train_rna, train_protein, train_labels = all_rna, all_protein, all_labels
            logging.warning('No validation file found, using all data for training')
    else:
        # Fallback: generate data if file doesn't exist
        logging.warning(f'Data file {data_file} not found, generating data...')
        train_rna, train_protein, train_labels = generate_dummy_data(
            num_samples=2000,
            rna_alphabet='AUCG',
            protein_alphabet='ARNDCEQGHILKMFPSTWYV',
            seed=42
        )
        val_rna, val_protein, val_labels = [], [], []
    
    # Load validation data from val.csv file if not already loaded
    if not val_rna and os.path.exists(val_file):
        from utils import load_data_from_csv
        logging.info(f'Loading validation data from {val_file}')
        val_rna, val_protein, val_labels = load_data_from_csv(val_file)
    elif not val_rna:
        # Fallback: split from training data if val.csv doesn't exist
        logging.warning(f'Validation file {val_file} not found, splitting from training data...')
        from sklearn.model_selection import train_test_split
        train_rna, val_rna, train_protein, val_protein, train_labels, val_labels = train_test_split(
            train_rna, train_protein, train_labels, test_size=0.1, random_state=42
        )
        # Save validation set
        from utils import save_data_to_csv
        save_data_to_csv(val_rna, val_protein, val_labels, val_file)
        logging.info(f'Saved validation set ({len(val_labels)} samples) to {val_file}')
    
    logging.info(f'Train set: {len(train_labels)} samples, Val set: {len(val_labels)} samples (from {val_file})')
    
    # Create datasets and loaders
    train_dataset = RNADataset(
        train_rna, train_protein, train_labels,
        max_rna_len=config['data']['max_rna_len'],
        max_protein_len=config['data']['max_protein_len']
    )
    val_dataset = RNADataset(
        val_rna, val_protein, val_labels,
        max_rna_len=config['data']['max_rna_len'],
        max_protein_len=config['data']['max_protein_len']
    )
    
    train_loader = create_data_loader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step(val_loss)  # ReduceLROnPlateau需要传入验证损失
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config
        }, is_best, checkpoint_dir)
        
        logging.info(f'Epoch {epoch+1} completed. Best Val Acc: {best_val_acc:.2f}%')
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logging.info(f'Early stopping at epoch {epoch+1}')
            break
    
    writer.close()
    logging.info('Training completed!')

if __name__ == '__main__':
    main()