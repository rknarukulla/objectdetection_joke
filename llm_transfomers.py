from transformers import AutoModel, AutoTokenizer

# Assuming the model is available as a local directory or on the Hugging Face Model Hub
model_name_or_path = "/Users/ravinarukulla/.cache/lm-studio/models/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q5_0.gguf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load the model
model = AutoModel.from_pretrained(model_name_or_path)

# Example usage
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(**inputs, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Replace "Your prompt here" with an actual prompt


if __name__ == '__main__':
    # Call the function and print the joke
    prompt = "tellme a joke"
    print(generate_text(prompt))
