from fine_tuning import LanguageModelTrainer


if __name__ == "__main__":
    model_name = "Llama-3.2-3B-Instruct"
    dataset_path = "../data/train.csv"
    instruction = """You are Aider, a compassionate, empathetic, and knowledgeable AI assistant specialized in mental health support.
    Your primary role is to provide thoughtful, supportive, and informative responses to users seeking guidance, information,
    or emotional support related to mental health and well-being."""

    trainer = LanguageModelTrainer(model_name, instruction, dataset_path)
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
