import pickle
import json

with open('user_patterns.pkl', 'rb') as f:
    data = pickle.load(f)

# Nếu data không phải là dict, hãy chuyển đổi cho phù hợp
with open('user_patterns.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2, default=str)  # default=str để tránh lỗi với object lạ