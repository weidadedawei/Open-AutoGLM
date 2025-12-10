from transformers import AutoTokenizer

model_path = "zai-org/AutoGLM-Phone-9B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]}
    ]
    
    # Try different apply_chat_template conventions
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("✅ Template output:")
        print(prompt)
    except Exception as e:
        print(f"❌ Template failed: {e}")
        
except Exception as e:
    print(f"❌ Tokenizer load failed: {e}")
