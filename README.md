# LLM_Pushback_Analysis
Chat Bot Pusha Analysis and evaluation


# Steps
1. Decide a chatbot, 
2. Create new conversations
3. Use existing conversations as templates with other chatbots.
# Pipeline
- [ ] Write a Python script or create a Colab notebook  
- [ ] Read prompts from a CSV file  
- [ ] Filter prompts based on configuration or user-provided arguments  
- [ ] Run the “user” prompt  
- [ ] Get the response from the “user” prompt  
- [ ] Run the “eval” prompt on all judge systems  
- [ ] Write to file for each row  
  - [ ] Blue = from dataset  
  - [ ] Purple = from system response  
  - [ ] Green = from judges  
- [ ] Include the following columns in the file:  
  - [ ] Prompt ID  
  - [ ] Prompt  
  - [ ] Error  
  - [ ] Error Type  
  - [ ] Error Salience  
  - [ ] Conversation Domain  
  - [ ] System Response  
  - [ ] System Name  
  - [ ] Error Acknowledge Response / Judge i  
  - [ ] Explanation Given Response / Judge i  
