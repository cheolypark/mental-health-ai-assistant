import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq


class LanguageModelTrainer:
    def __init__(self, model_name, instruction, dataset_path, max_seq_length=2048, dtype=None, load_in_4bit=True):
        self.model_name = model_name
        self.instruction = instruction
        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None

    def check_gpu(self):
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

    def load_model_and_tokenizer(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name='unsloth/' + self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

    def prepare_dataset(self):
        dataset = load_dataset('csv', data_files=self.dataset_path)

        def format_chat_template(row):
            row_json = [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": row["Context"]},
                {"role": "assistant", "content": row["Response"]}
            ]
            row["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
            return row

        dataset = dataset["train"].map(format_chat_template)
        self.dataset = dataset.train_test_split(test_size=0.1)

    def setup_peft_model(self, r=16, lora_alpha=16, lora_dropout=0, target_modules=None, random_state=3407):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=random_state,
            use_rslora=False,
            loftq_config=None,
        )

    def train_model(self, output_dir="model_training_outputs",
                    per_device_train_batch_size=2, per_device_eval_batch_size=2,
                    gradient_accumulation_steps=4, eval_steps=0.2, warmup_steps=5,
                    max_steps=60, learning_rate=2e-4, weight_decay=0.01, seed=3407):

        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
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
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            num_proc=1,
        )

        trainer_stats = self.trainer.train()
        print(trainer_stats)

    def inference(self, user_input, max_new_tokens=150):
        FastLanguageModel.for_inference(self.model)

        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_input}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = text.split("assistant")[-1].strip()
        return response

    def save_model_and_tokenizer(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
