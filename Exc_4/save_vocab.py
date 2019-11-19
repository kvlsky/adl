from collections import Counter
import pickle


with open('Exc_4/tinyshakespeare.txt') as text_file:
    data = text_file.read()
    num_chars = print(len(data.split()))

vocab = Counter(data).most_common()
print(vocab)

dict = {}

'''
beacause counter is already sorted (desc) by character frequency,
we do not use moset_common() method
'''
with open('Exc_4/vocabulary.pkl', 'wb+') as vocab_file:
    idx2char = {i: u[0] for i, u in enumerate(vocab)}
    print(idx2char)
    pickle.dump(idx2char, vocab_file, pickle.HIGHEST_PROTOCOL)
