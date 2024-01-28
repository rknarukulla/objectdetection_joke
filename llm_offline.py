from transformers import GPT2LMHeadModel, GPT2Tokenizer


def tell_a_joke():
    # Load pre-downloaded GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Prompt for the model to generate a joke
    # prompt = "Tell me a joke about 5 persons"

    # Encode the prompt to be compatible with the model
    # input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #
    # # Generate a response
    # output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    #
    # # Decode the generated text
    # joke = tokenizer.decode(output[0], skip_special_tokens=True)

    from transformers import pipeline  # Example using Hugging Face

    generator = pipeline("text-generation", model="gpt2", truncation=True)  # Replace with desired model

    context = ("Imagine, You are looking at a scene and you see 5 people,and 1 phone.  Give a complement using current time and scene")

    joke = generator(context, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]

    joke = joke.replace(context, "").strip()  # Remove context and leading/trailing whitespace

    # print("Joke:", joke)

    return joke


if __name__ == '__main__':
    # Call the function and print the joke
    print(tell_a_joke())
