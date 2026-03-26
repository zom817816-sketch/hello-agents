# BPE分词算法模拟 
import re, collections 

def get_stats(vocab): 
    """
    统计词元对频率
    """ 
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items(): 
        symbols = word.split() 
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq 
    return pairs 

def merge_vocab(pair, v_in): 
    """
    合并词元对
    """ 
    v_out = {} 
    bigram = re.escape(" ".join(pair)) 
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)") 
    for word in v_in: 
        w_out = p.sub("".join(pair), word) 
        v_out[w_out] = v_in[word] 
    return v_out 

# 准备语料库,在每个词末尾加上</w>表示结束,并且分好字符
vocab = {"h u b </w>": 1, "p u g </w>": 1, "p u n </w>": 1, "b u n </w>": 1} 
num_merges = 4 # 设置合并次数 

for i in range(num_merges): 
    pairs = get_stats(vocab) 
    if not pairs: 
        break 
    best = max(pairs, key = pairs.get) 
    vocab = merge_vocab(best, vocab) 
    print(f"第{i+1}次合并: {best} -> {''.join(best)}") 
    print(f"新词表(部分): {list(vocab.keys())}")
    print('-' * 20)

# 配置环境和选择模型 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型地址
model_path = "C:\\Users\\MOMO\\Desktop\\Qwen3.5-0.8B" 

# 设置设备 
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"Using device: {device}") 

# 加载分词器 
tokenizer = AutoTokenizer.from_pretrained(model_path) 

# 加载模型 
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

print('模型和分词器加载完成') 

# 准备对话输入 
messages = [
{'role': 'system', 'content': 'You are a helpful assistant'}, 
{'role': 'user', 'content': '请介绍你自己'}
]

# 使用分词器模板格式化输入
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 编码输入文本 
model_inputs = tokenizer([text], return_tensors='pt').to(device) 

print(f"编码后的输入文本:")
print(model_inputs) 

# 使用模型生成回答 
# max_new_tokens控制模型最多生成多少个新的token 
generated_ids = model.generate(
    model_inputs.input_ids, 
    max_new_tokens=512
)

# 将生成的Token ID 截取输入部分 
# 这样我们只解码模型新生成的部分 
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) # zip对batch中的每条输入和输出配对,不同对的输入长度不同
]

# 解码生成的Token ID 
response = tokenzier.batch_decode(generated_ids, skip_special_tokens=True)[0] 

print('\n模型的回答') 
print(response) 

