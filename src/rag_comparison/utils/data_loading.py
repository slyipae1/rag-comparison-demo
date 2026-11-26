import json

# later standardize the data format
def load_data(path):
    
    text_data = []

    if path.endswith('.json'):
        data = json.load(open(path, 'r', encoding='utf-8'))
    elif path.endswith('.jsonl'):
        data = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    
    for item in data:
        if isinstance(item, dict):
            text_data.append(item.get('document', ''))
        elif isinstance(item, str):
            text_data.append(item)
        else:
            raise ValueError("Unsupported data format. Each item should be a string or a dictionary with a 'document' key.")
        
    return text_data
