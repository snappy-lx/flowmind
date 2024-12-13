import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50Tokenizer
from datasets import Dataset
import xml.etree.ElementTree as ET
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from loguru import logger
import os
import argparse
import gc
from tqdm import tqdm


class TranslationTrainer:
    def __init__(self, model_name="facebook/mbart-large-50", source_lang="en_XX", target_lang="zh_CN"):
        """
        Initialize the translation trainer with specified model and languages
        
        Args:
            model_name (str): Name of the pretrained model
            source_lang (str): Source language code
            target_lang (str): Target language code
        """
        # Set up device - use MPS for M-chip Macs if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Map model names to tokenizer classes
        tokenizer_classes = {
            "facebook/mbart-large-50": MBart50Tokenizer,
            # Add other model-tokenizer mappings here if needed
        }

        # Determine the tokenizer class based on the model name
        tokenizer_class = tokenizer_classes.get(model_name, MBartTokenizer)

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.tokenizer.src_lang = source_lang
        self.tokenizer.tgt_lang = target_lang
        
        # Initialize model
        logger.info(f"Loading model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
    
    def validate_text(self, text, idx):
        """
        Validate text entries for proper format and content
        
        Args:
            text (str): Text to validate
            idx (int): Index for logging purposes
        
        Returns:
            bool: True if text is valid, False otherwise
        """
        if text is None or not text.strip():
            return False

        if not isinstance(text, str):
            logger.warning(f"Entry {idx}: Invalid type - {type(text)}: {text}")
            return False
        
        # Add more validation rules as needed
        return True
    
    def parse_tmx(self, tmx_path):
        """
        Parse TMX file and extract translation pairs with validation
        
        Args:
            tmx_path (str): Path to TMX file
        
        Returns:
            pd.DataFrame: DataFrame containing valid translation pairs
        """
        logger.info(f"Parsing TMX file: {tmx_path}")
        
        tree = ET.parse(tmx_path)
        root = tree.getroot()
        
        source_texts = []
        target_texts = []
        skipped_count = 0
        total_count = 0
        
        # Process translation units with progress bar
        for tu in root.findall('.//tu'):
            total_count += 1
            tuv_elements = tu.findall('.//tuv')
            
            if len(tuv_elements) != 2:
                skipped_count += 1
                continue
                
            try:
                # Get language attributes
                lang1 = tuv_elements[0].get('{http://www.w3.org/XML/1998/namespace}lang', '')
                lang2 = tuv_elements[1].get('{http://www.w3.org/XML/1998/namespace}lang', '')
                
                # Extract text
                text1 = tuv_elements[0].find('.//seg').text
                text2 = tuv_elements[1].find('.//seg').text
                
                # Validate texts
                if not (self.validate_text(text1, total_count) and self.validate_text(text2, total_count)):
                    skipped_count += 1
                    continue
                
                # Ensure correct language order
                if lang1.startswith('en') and lang2.startswith('zh'):
                    source, target = text1, text2
                elif lang2.startswith('en') and lang1.startswith('zh'):
                    source, target = text2, text1
                else:
                    logger.warning(f"Entry {total_count}: Invalid language pair - {lang1}, {lang2}")
                    skipped_count += 1
                    continue
                
                source_texts.append(source)
                target_texts.append(target)
                
            except (AttributeError, TypeError) as e:
                logger.warning(f"Entry {total_count}: Error processing entry - {str(e)}")
                skipped_count += 1
                continue
        
        # Create DataFrame
        df = pd.DataFrame({
            'source_text': source_texts,
            'target_text': target_texts
        })
        
        # Log statistics
        logger.info(f"Total entries processed: {total_count}")
        logger.info(f"Valid pairs extracted: {len(df)}")
        logger.info(f"Skipped entries: {skipped_count}")
        logger.info(f"Success rate: {(len(df)/total_count)*100:.2f}%")
        
        return df
    
    def prepare_dataset(self, df, max_length=256):
        """
        Prepare dataset for training with efficient memory handling
        
        Args:
            df (pd.DataFrame): Input DataFrame with translation pairs
            max_length (int): Maximum sequence length
        
        Returns:
            Dataset: HuggingFace dataset ready for training
        """
        logger.info("Preparing dataset for training...")
        
        # Convert to strings and clean
        df['source_text'] = df['source_text'].astype(str)
        df['target_text'] = df['target_text'].astype(str)
        
        dataset = Dataset.from_pandas(df)
        
        def preprocess_function(examples):
            # Tokenize source texts
            model_inputs = self.tokenizer(
                examples['source_text'],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize target texts
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['target_text'],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
            
            return {
                'input_ids': model_inputs['input_ids'].squeeze(0),
                'attention_mask': model_inputs['attention_mask'].squeeze(0),
                'labels': labels['input_ids'].squeeze(0)
            }
        
        # Process dataset with batching for memory efficiency
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc="Tokenizing texts"
        )
        
        dataset.set_format(type='torch')
        return dataset
    
    def train(self, dataset, num_epochs=3, batch_size=16, learning_rate=5e-5, gradient_accumulation_steps=8):
        """
        Train the model with memory-efficient settings
        
        Args:
            dataset: Prepared dataset
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
        """
        logger.info("Starting training...")
        
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Set up learning rate scheduler
        total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
        warmup_steps = total_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Clear cache if possible
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate loss
                    loss = outputs.loss / gradient_accumulation_steps
                    current_loss = loss.item()
                    total_loss += current_loss * gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': current_loss})
                    
                    # Clean up memory
                    del outputs, loss
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    # Update weights if needed
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM in batch {batch_idx}. Attempting recovery...")
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        optimizer.zero_grad()
                        continue
                    raise e
            
            # End of epoch
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch+1}")
            
            # Force garbage collection
            gc.collect()
    
    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train translation model using TMX data')
    parser.add_argument('--tmx_path', type=str, required=True,
                      help='Path to the TMX file containing translation pairs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                      help='Maximum sequence length')
    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-50",
                      help='Pretrained model name')
    parser.add_argument('--source_lang', type=str, default="en_XX",
                      help='Source language code')
    parser.add_argument('--target_lang', type=str, default="zh_CN",
                      help='Target language code')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TranslationTrainer(
        model_name=args.model_name,
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )
    
    # Process data
    df = trainer.parse_tmx(args.tmx_path)
    print(df.head())  # Print the first few rows of the DataFrame to validate parsing

    dataset = trainer.prepare_dataset(df, max_length=args.max_length)
    
    # Train model
    trainer.train(
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
