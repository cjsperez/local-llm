import json

with open('brands.json') as f:
    brand_configs = json.load(f)

def build_prompt(brand_key, customer_name=None):
    config = brand_configs[brand_key]
    greeting = f"Hi {customer_name}!" if customer_name else "Hello!"
    
    return f"""
{greeting}

Brand: {config['display_name']}
Products: {', '.join(config['products'])}

Rules:
1. Fix known name issues (e.g., {"; ".join(f"{k} → {v}" for k, v in config['corrections'].items())})
2. Do not answer questions outside of brand scope.
3. If unsure or off-topic, say: "{config['off_topic_response']}"
4. Use a maximum of {config['word_limit']} words.
5. For more help, refer to: {config['support_email']}
6. Use conversational, spoken tone (no markdown, no special characters).
"""
