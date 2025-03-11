from transformers import pipeline
import json

# Load the text generation pipeline
pipeline = pipeline("text-generation", model="google/gemma-2-2b-it")

# Start the conversation and store exchanges in a list
conversation = []
user_input = "Hello, how are you today?"

# Loop for conversation generation
while user_input.lower() not in ["exit", "quit", "bye"]:
    # Add the user input to the conversation
    conversation.append({"role": "user", "content": user_input})
    
    # Generate a response from the model
    prompt = pipeline.tokenizer.apply_chat_template(conversation,
                                                tokenize=False,
                                                add_generation_prompt=True)
    response = pipeline(prompt, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    print(response)
    text = response[0]["generated_text"]
    text = text.replace(prompt, '', 1)
    
# Save the conversation as a JSON file
with open("conversation.json", "w") as json_file:
    json.dump(conversation, json_file, indent=4)
