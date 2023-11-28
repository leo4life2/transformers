import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from torchvision.transforms import Resize
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    FuyuProcessor,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    BitsAndBytesConfig
)

import transformers
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Constants (hardcoded values for CLI arguments)
DATASET_NAME = "facebook/winoground"
MODEL_NAME_OR_PATH = "adept/fuyu-8b"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
BLOCK_SIZE = 128
OUTPUT_DIR = "./tmp/test-fuyu"
VALIDATION_SPLIT_PERCENTAGE = 5
SEED = 42
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
NUM_TRAIN_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER_TYPE = SchedulerType.LINEAR
NUM_WARMUP_STEPS = 0
CHECKPOINTING_STEPS = None
RESUME_FROM_CHECKPOINT = None
WITH_TRACKING = False
REPORT_TO = "all"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if SEED is not None:
        set_seed(SEED)

    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets
    raw_datasets = load_dataset(DATASET_NAME)
    if DATASET_NAME == "facebook/winoground":
        raw_datasets["validation"] = load_dataset(
            DATASET_NAME,
            split=f"test[:{VALIDATION_SPLIT_PERCENTAGE}%]",
        )
        raw_datasets["validation"] = raw_datasets["validation"].select(range(10))
        raw_datasets["train"] = load_dataset(
            DATASET_NAME,
            split=f"test[{VALIDATION_SPLIT_PERCENTAGE}%:]",
        )
        raw_datasets["train"] = raw_datasets["train"].select(range(50))
        raw_datasets["test"] = raw_datasets["validation"]
    elif "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            DATASET_NAME,
            split=f"train[:{VALIDATION_SPLIT_PERCENTAGE}%]",
        )
        raw_datasets["train"] = load_dataset(
            DATASET_NAME,
            split=f"train[{VALIDATION_SPLIT_PERCENTAGE}%:]",
        )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load image_processor
    processor = FuyuProcessor.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer
    processor.max_position_embeddings = BLOCK_SIZE
    tokenizer.model_max_length = BLOCK_SIZE
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        config=config,
        quantization_config=bnb_config,
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        target_modules=["query_key_value"],
        init_lora_weights=False
    )
    model.add_adapter(lora_config, adapter_name="lora")
    model.tie_weights()

    # Resize embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets
    column_names = raw_datasets["train"].column_names
    text_column_name = "caption_0" if "caption_0" in column_names else column_names[0]
    image_column_name = "image_0" if "image_0" in column_names else column_names[1]

    def filter_corrupt_images(examples):
        valid_images = []
        for i in range(0, len(examples[image_column_name])):
            try:
                processor(text="test", images=[examples[image_column_name][i]])
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    with accelerator.main_process_first():
        raw_datasets = raw_datasets.filter(filter_corrupt_images, batched=True)
        resize = Resize((224, 224))
        raw_datasets = raw_datasets.map(lambda x: {"image_0": resize(x["image_0"])}, batched=False)

    block_size = min(BLOCK_SIZE, tokenizer.model_max_length)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    # Log a few random samples from the training set
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def collate_fn(examples):
        texts = [e[text_column_name] for e in examples]
        images = [e[image_column_name] for e in examples]
        output = processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True
        )
        position = (output["input_ids"] == tokenizer.vocab["<s>"]).nonzero(as_tuple=True)[0][0]
        output["labels"] = torch.full_like(output["input_ids"], -100)
        output["labels"][position:] = output["input_ids"][position:]
        return output

    # DataLoaders creation
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=PER_DEVICE_EVAL_BATCH_SIZE)

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # Scheduler and math around the number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE.value,
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=max_train_steps * GRADIENT_ACCUMULATION_STEPS,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    if CHECKPOINTING_STEPS is not None and CHECKPOINTING_STEPS.isdigit():
        checkpointing_steps = int(CHECKPOINTING_STEPS)
    else:
        checkpointing_steps = None

    # Train!
    total_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {PER_DEVICE_TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = RESUME_FROM_CHECKPOINT
        path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * GRADIENT_ACCUMULATION_STEPS
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // GRADIENT_ACCUMULATION_STEPS
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, NUM_TRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if checkpointing_steps and completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                if OUTPUT_DIR is not None:
                    output_dir = os.path.join(OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)
            if completed_steps >= max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(PER_DEVICE_EVAL_BATCH_SIZE)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if CHECKPOINTING_STEPS == "epoch":
            output_dir = f"epoch_{epoch}"
            if OUTPUT_DIR is not None:
                output_dir = os.path.join(OUTPUT_DIR, output_dir)
            accelerator.save_state(output_dir)

    if OUTPUT_DIR is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(OUTPUT_DIR)

            with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

            peak_memory = torch.cuda.max_memory_allocated("cuda")
            print(f"Peak VRAM usage: {peak_memory / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
