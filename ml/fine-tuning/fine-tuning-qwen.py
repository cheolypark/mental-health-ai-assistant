from fine_tuning import LanguageModelTrainer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


class LanguageModelTrainerQwen(LanguageModelTrainer):
    def __init__(self, model_name, instruction, dataset_path):
        super().__init__(model_name, instruction, dataset_path)

    def prepare_dataset(self):
        dataset = load_dataset('csv', data_files=self.dataset_path, split="train")

        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN

        def formatting_prompts_func(examples):
            inputs = examples['Context']
            outputs = examples['Response']
            instructions = [self.instruction] * len(inputs)
            texts = []
            for instr, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instr, input, output) + EOS_TOKEN
                texts.append(text)

            return {"text": texts}

        self.dataset = dataset.map(formatting_prompts_func, batched=True)

    def train_model(self, output_dir="model_training_outputs",
                    per_device_train_batch_size=2, per_device_eval_batch_size=2,
                    gradient_accumulation_steps=4, eval_steps=0.2, warmup_steps=5,
                    max_steps=60, learning_rate=2e-4, weight_decay=0.01, seed=3407):

        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps = 1,
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=output_dir,
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field = "text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

        trainer_stats = self.trainer.train()
        print(trainer_stats)

    def inference(self, user_input, max_new_tokens=150):
        FastLanguageModel.for_inference(self.model)

        inputs = self.tokenizer(
            [
                alpaca_prompt.format(
                    self.instruction, # instruction
                    user_input, # input
                    "", # output - leave this blank for generation!
                )
            ], return_tensors = "pt").to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True)
        response = self.tokenizer.batch_decode(outputs)
        return response


if __name__ == "__main__":
    model_name = "Qwen2.5-7B"
    dataset_path = "../data/train.csv"
    instruction = """You are Aider, a compassionate, empathetic, and knowledgeable AI assistant specialized in mental health support.
    Your primary role is to provide thoughtful, supportive, and informative responses to users seeking guidance, information,
    or emotional support related to mental health and well-being."""

    trainer = LanguageModelTrainerQwen(model_name, instruction, dataset_path)
    trainer.check_gpu()
    trainer.load_model_and_tokenizer()
    trainer.prepare_dataset()
    trainer.setup_peft_model()
    trainer.train_model(output_dir="../outputs/" + model_name)

    # Inference example
    user_input = ("I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. "
                  "How can I change my feeling of being worthless to everyone?")
    response = trainer.inference(user_input)
    print("Assistant Response:", response)

    # Save trained model
    trainer.save_model_and_tokenizer("../models/" + model_name)
