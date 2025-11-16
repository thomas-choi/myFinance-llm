import argparse
from transformers import AutoModelForTimeSeriesPrediction, AutoProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/finetuned')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # Load dataset (windows as sequences, targets as next OHLCV for forecasting)
    dataset = load_dataset('csv', data_files=args.data)
    # Preprocess: adapt windows to inputs, labels to next day OHLCV (not classification; forecast then derive)
    # For simplicity, assume data_prep provides 'sequence' and 'target' columns for next OHLCV
    # TODO: Modify data_prep if needed for forecasting targets

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForTimeSeriesPrediction.from_pretrained(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],  # Split inside
    )
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()
