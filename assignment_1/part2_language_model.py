# ==========================================================
# Import Libraries
# ==========================================================

import random
import string
from collections import Counter, defaultdict

import nltk
import math
import pandas as pd

# ==========================================================
# Download Dataset
# ==========================================================
nltk.download("brown")

# ==========================================================
# Load Corpus
# ==========================================================
from nltk.corpus import brown

print("=" * 60)
print("Loading Brown Corpus...")
print("=" * 60)

sentences = brown.sents()

print(f"Total Sentences : {len(sentences):,}")

# ==========================================================
# Preprocess Sentences
# ==========================================================
processed_sentences = []

for sentence in sentences:

    current_sentence = []

    for token in sentence:

        token = token.lower()
        token = token.strip(string.punctuation)

        if token != "":
            current_sentence.append(token)

    if len(current_sentence) > 0:
        processed_sentences.append(current_sentence)

print(f"Processed Sentences : {len(processed_sentences):,}")

# ==========================================================
# Train Test Split
# ==========================================================
random.seed(42)

random.shuffle(processed_sentences)

split_index = int(
    0.8 * len(processed_sentences)
)

train_sentences = processed_sentences[:split_index]

test_sentences = processed_sentences[split_index:]

print(f"Training Sentences : {len(train_sentences):,}")
print(f"Testing Sentences  : {len(test_sentences):,}")

# ==========================================================
# Add Sentence Tokens
# ==========================================================
train_data = []

for sentence in train_sentences:

    train_data.append(
        ["<s>"] + sentence + ["</s>"]
    )

test_data = []

for sentence in test_sentences:

    test_data.append(
        ["<s>"] + sentence + ["</s>"]
    )

print("Sentence boundary tokens added.")

# ==========================================================
# Build Vocabulary
# ==========================================================
vocabulary = set()

for sentence in train_data:

    vocabulary.update(sentence)

print(f"Vocabulary Size : {len(vocabulary):,}")

# ==========================================================
# Unigram Count
# ==========================================================
unigram_counts = Counter()

for sentence in train_data:

    unigram_counts.update(sentence)

total_tokens = sum(unigram_counts.values())

print(f"Total Training Tokens : {total_tokens:,}")

# ==========================================================
# Bigram Count
# ==========================================================
bigram_counts = Counter()

for sentence in train_data:

    for i in range(len(sentence) - 1):

        bigram = (
            sentence[i],
            sentence[i + 1]
        )

        bigram_counts[bigram] += 1

print(f"Unique Bigrams : {len(bigram_counts):,}")

# ==========================================================
# Q6 - Unigram Probabilities
# ==========================================================
unigram_probability = {}

for word in unigram_counts:

    unigram_probability[word] = (
        unigram_counts[word] /
        total_tokens
    )

print("Unigram model created.")

# ==========================================================
# Bigram Probabilities
# ==========================================================

bigram_probability = {}

for (previous_word, current_word), count in bigram_counts.items():

    bigram_probability[
        (previous_word, current_word)
    ] = (
        count /
        unigram_counts[previous_word]
    )

print("Bigram model created.")

# ==========================================================
# Sample output
# ==========================================================
print("\nSample Unigram Probabilities\n")

for word in list(unigram_probability.keys())[:10]:

    print(
        f"{word:15s} {unigram_probability[word]:.6f}"
    )

print("\nSample Bigram Probabilities\n")

for bigram in list(bigram_probability.keys())[:10]:

    print(
        f"{bigram} -> {bigram_probability[bigram]:.6f}"
    )


# ==========================================================
#  Q7 - Laplace Smoothing
# ==========================================================
print("\n" + "=" * 60)
print("Q7 : Laplace Smoothing")
print("=" * 60)

def laplace_bigram_probability(
    previous_word,
    current_word
):

    bigram_count = bigram_counts.get(
        (previous_word, current_word),
        0
    )

    unigram_count = unigram_counts.get(
        previous_word,
        0
    )

    vocabulary_size = len(vocabulary)

    probability = (bigram_count + 1) / (unigram_count + vocabulary_size)

    return probability

# ==========================================================
#  Compare Probabilities
# ==========================================================
seen_bigram = (
    "of",
    "the"
)

unseen_bigram = (
    "elephant",
    "computer"
)

seen_probability = laplace_bigram_probability(
    seen_bigram[0],
    seen_bigram[1]
)

unseen_probability = laplace_bigram_probability(
    unseen_bigram[0],
    unseen_bigram[1]
)

mle_seen = bigram_probability.get(seen_bigram, 0)

mle_unseen = bigram_probability.get(unseen_bigram, 0)

print("\nMLE Probabilities")

print(f"P({seen_bigram[1]} | {seen_bigram[0]}) = {mle_seen:.8f}")
print(f"P({unseen_bigram[1]} | {unseen_bigram[0]}) = {mle_unseen:.8f}")

print("\nLaplace Smoothed Probabilities")

print(f"P({seen_bigram[1]} | {seen_bigram[0]}) = {seen_probability:.8f}")
print(f"P({unseen_bigram[1]} | {unseen_bigram[0]}) = {unseen_probability:.8f}")

# ==========================================================
#  Observation
# ==========================================================

print("\nObservation:")

print("Without smoothing, unseen bigrams receive")

print("zero probability.")

print()

print("Laplace smoothing assigns a small")

print("non-zero probability to unseen")

print("word pairs.")

print()

print("This prevents the language model")

print("from assigning zero probability")

print("to an entire sentence.")

# ==========================================================
#  Q8 - Perplexity
# ==========================================================

print("\n" + "=" * 60)
print("Q8 : Perplexity Evaluation")
print("=" * 60)


def sentence_log_probability_mle(sentence):

    log_probability = 0.0

    for i in range(len(sentence) - 1):

        bigram = (
            sentence[i],
            sentence[i + 1]
        )

        probability = bigram_probability.get(
            bigram,
            0
        )

        if probability == 0:

            return float("-inf")

        log_probability += math.log(probability)

    return log_probability

# ==========================================================
#  Laplace Sentence Probability
# ==========================================================
def sentence_log_probability_laplace(sentence):

    log_probability = 0.0

    for i in range(len(sentence) - 1):

        probability = laplace_bigram_probability(

            sentence[i],

            sentence[i + 1]

        )

        log_probability += math.log(probability)

    return log_probability

# ==========================================================
#  Perplexity Function
# ==========================================================
def compute_perplexity(
    dataset,
    probability_function
):

    total_log_probability = 0.0

    total_words = 0

    skipped_sentences = 0

    for sentence in dataset:

        log_probability = probability_function(sentence)

        if log_probability == float("-inf"):

            skipped_sentences += 1
            continue

        total_log_probability += log_probability

        total_words += len(sentence) - 1

    if total_words == 0:

        return float("inf")

    perplexity = math.exp(

        -total_log_probability / total_words

    )

    print(f"Skipped Sentences : {skipped_sentences}")

    return perplexity

# ==========================================================
#  Evaluate
# ==========================================================
mle_perplexity = compute_perplexity(

    test_data,

    sentence_log_probability_mle

)

laplace_perplexity = compute_perplexity(

    test_data,

    sentence_log_probability_laplace

)

print(f"\nMLE Perplexity      : {mle_perplexity:.2f}")

print(f"Laplace Perplexity : {laplace_perplexity:.2f}")

# ==========================================================
#  Observation
# ==========================================================
print("\nObservation:")

print("The MLE model skipped many test sentences because unseen")
print("bigrams receive zero probability.")

print()

print("The Laplace-smoothed model evaluated every test sentence")
print("by assigning non-zero probabilities to unseen bigrams.")

print()

print("Although the MLE perplexity appears lower, it was computed")
print("on a much smaller subset of the test data.")

print()

print("Therefore, Laplace smoothing provides a more robust")
print("language model for unseen text.")

# ==========================================================
#  Q9 - Sentence Generation
# ==========================================================
print("\n" + "=" * 60)
print("Q9 : Sentence Generation")
print("=" * 60)

bigram_lookup = defaultdict(list)

for (previous_word, current_word), probability in bigram_probability.items():
    bigram_lookup[previous_word].append((current_word, probability))


# ==========================================================
# Generate Sentence
# ==========================================================

def generate_sentence(max_length=15):

    sentence = []
    current_word = "<s>"

    while len(sentence) < max_length:
        if current_word not in bigram_lookup:
            break

        next_words = []
        probabilities = []

        for word, probability in bigram_lookup[current_word]:

            next_words.append(word)
            probabilities.append(probability)

        current_word = random.choices(

            next_words,
            weights=probabilities,
            k=1

        )[0]

        if current_word == "</s>":

            break

        sentence.append(current_word)

    return " ".join(sentence)

# ==========================================================
# Generate 5 Sentences
# ==========================================================
print("\nGenerated Sentences\n")

random.seed(42)
for i in range(5):
    sentence = generate_sentence()
    print(f"{i+1}. {sentence}")

# ==========================================================
# Q10 - Trigram Language Model
# ==========================================================
print("\n" + "=" * 60)
print("Q10 : Trigram Language Model")
print("=" * 60)

trigram_counts = Counter()

for sentence in train_data:

    for i in range(len(sentence) - 2):

        trigram = (
            sentence[i],
            sentence[i + 1],
            sentence[i + 2]
        )

        trigram_counts[trigram] += 1

print(f"Unique Trigrams : {len(trigram_counts):,}")

# ==========================================================
# Trigram Probabilities
# ==========================================================

trigram_probability = {}

for trigram, count in trigram_counts.items():

    previous_bigram = (
        trigram[0],
        trigram[1]
    )

    previous_count = bigram_counts[previous_bigram]

    trigram_probability[trigram] = (
        count / previous_count
    )

print("Trigram model created.")

# ==========================================================
# Trigram Sentence Probability
# ==========================================================

def sentence_log_probability_trigram(sentence):

    log_probability = 0.0

    for i in range(len(sentence) - 2):

        trigram = (
            sentence[i],
            sentence[i + 1],
            sentence[i + 2]
        )

        probability = trigram_probability.get(
            trigram,
            0
        )

        if probability == 0:

            return float("-inf")

        log_probability += math.log(probability)

    return log_probability

# ==========================================================
# TRIGRAM PERPLEXITY
# ==========================================================

def compute_trigram_perplexity(dataset):

    total_log_probability = 0.0

    total_predictions = 0

    skipped_sentences = 0

    for sentence in dataset:

        log_probability = sentence_log_probability_trigram(sentence)

        if log_probability == float("-inf"):

            skipped_sentences += 1
            continue

        total_log_probability += log_probability

        # Number of trigram predictions in one sentence
        total_predictions += len(sentence) - 2

    print(f"Skipped Sentences : {skipped_sentences}")

    if total_predictions == 0:

        return float("inf")

    perplexity = math.exp(
        -total_log_probability / total_predictions
    )

    return perplexity


# ==========================================================
# EVALUATE TRIGRAM MODEL
# ==========================================================

trigram_perplexity = compute_trigram_perplexity(
    test_data
)

print(f"\nTrigram Perplexity : {trigram_perplexity:.2f}")

# ==========================================================
# Comparison
# ==========================================================
print("\nComparison")

print(f"Bigram MLE Perplexity : {mle_perplexity:.2f}")

print(f"Trigram Perplexity    : {trigram_perplexity:.2f}")

# ==========================================================
# OBSERVATION
# ==========================================================

print("\nObservation:")

print("The trigram model uses the previous two words to predict")
print("the next word, providing more context than the bigram model.")

print()
print("However, many test sentences contain unseen trigrams.")

print()
print("As a result, a large number of sentences were skipped")
print("during evaluation because the unsmoothed trigram model")
print("assigned zero probability to unseen trigrams.")
print()
print("The reported trigram perplexity is therefore calculated")
print("only on the remaining evaluated sentences.")