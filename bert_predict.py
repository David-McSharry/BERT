from BertModules.BertConfig import BertConfig
import transformers
from utils import load_pretrained_weights, predict

config = BertConfig()
my_bert = load_pretrained_weights(config)
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

your_text = "The capital of [MASK] is Paris."
predictions = predict(my_bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))
predicted_sentence = your_text.replace("[MASK]", predictions[0][0])
print("Completed sentence with top result:", predicted_sentence)

