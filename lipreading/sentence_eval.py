from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def beam_search(model, tokenizer, possible_words, beam_width=20, k=5):
	beam_candidates = []
	for i in range(len(possible_words[0])):
		for j in range(len(possible_words[1])):
			candidate = possible_words[0][i] + ' ' + possible_words[1][j]
			beam_candidates.append((sent_scoring(model, tokenizer, candidate), candidate))
	beam_candidates.sort(reverse=True)
	beam_candidates = beam_candidates[:beam_width]

	for pos in range(2, len(possible_words)):
		new_candidates = []
		for candidate in beam_candidates:
			for i in range(len(possible_words[pos])):
				new_candidate = candidate[1] + ' ' + possible_words[pos][i]
				new_candidates.append((sent_scoring(model, tokenizer, new_candidate), new_candidate))
		new_candidates.sort(reverse=True)
		beam_candidates = new_candidates[:beam_width]

	return [beam_candidates[i][1] for i in range(k)]

def sent_scoring(model, tokenizer, sentence):
	model.eval()

	input_ids = tokenizer.encode(sentence, return_tensors="pt")

	with torch.no_grad():
		output = model(input_ids)[0]
		log_likelihood = torch.log_softmax(output, dim=-1).squeeze()[1]

	return log_likelihood

def evaluate_sentence(model, X_test, Y_test, sentence_start_idx, vocab_list):
	distilbert_tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-CoLA")
	distilbert_model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-CoLA")

	output_probs = model(X_test)
	top_k_values, top_k_indices = tf.nn.top_k(output_probs, k=5)
	top_k_indices = np.array(top_k_indices)
	correct = 0
	total = len(sentence_start_idx)
	for i, idx in enumerate(sentence_start_idx):
		next_idx = sentence_start_idx[i+1] if i+1<len(sentence_start_idx) else len(Y_test)
		possible_words = {}
		for word_position in range(next_idx-idx):
			top_k_preds = top_k_indices[word_position+idx]
			possible_words[word_position] = [vocab_list[pred] for pred in top_k_preds]
		candidate_sentences = beam_search(distilbert_model, distilbert_tokenizer, possible_words)
		real_sentence = [vocab_list[Y_test[i][0]] for i in range(idx, next_idx)] 
        
        if real_sentence in candidate_sentences:
            correct += 1

    return correct/total