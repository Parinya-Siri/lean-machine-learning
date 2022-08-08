from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import utilities.utilities as util


inp = 'print("Hello world!")'

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')

tokenizer.add_special_tokens ({
"eos_token": "</s>",
"bos_token": "<s>",
"unk_token": "<unk>",
"pad_token": "<pad>",
"mask_token": "<mask>"
})

t = tokenizer.encode(inp)

print(t)
print (tokenizer.decode(t))

model = GPT2LMHeadModel.from_pretrained("GPyT").to("cuda")

while True:
    inp = input(">>>")
    #inp inp.replace()
    input_ids = tokenizer.encode(inp, return_tensors="pt").to("cuda")
    beam_output = model.generate(
        input_ids,
        max_length = 102,
        num_beams = 102,
        temperature = 0.7,
        no_repeat_ngram_size = 2,
        num_return_sequences = 1)
    for beam in beam_output:
        out = tokenizer.decode(beam)
        fout = out.replace("<N>","\n")
        print (str(fout))