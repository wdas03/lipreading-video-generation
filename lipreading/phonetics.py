import jellyfish

def create_phonetics(vocab_list)
    phonetic_set = set()
    word_to_phonetic = {}
    phonetic_to_word = defaultdict(list)
    for word in vocab_list:
        phonetic_set.add(jellyfish.soundex(word))
        word_to_phonetic[word] = jellyfish.soundex(word)
        phonetic_to_word[jellyfish.soundex(word)].append(word)
    phonetic_list = sorted(list(phonetic_set))

    phonetic_to_label = {p:i for i,p in enumerate(phonetic_list)}
    label_to_phonetic = {i:p for i,p in enumerate(phonetic_list)}
    return phonetic_to_label, label_to_phonetic

def word_label_to_phonetic_label(x):
    word = vocab_list[x]
    phonetic = word_to_phonetic[word]
    label = phonetic_to_label[phonetic]
    return label