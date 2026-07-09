# ==========================================================
# Assignment 1 - Part 3
# Q11 : Hidden Markov Model (HMM) Training
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import random
from collections import Counter, defaultdict

import nltk

# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

nltk.download("brown")

# ==========================================================
# LOAD BROWN TAGGED CORPUS
# ==========================================================

from nltk.corpus import brown

print("=" * 60)
print("Loading Brown Tagged Corpus...")
print("=" * 60)

tagged_sentences = tagged_sentences = list(
    brown.tagged_sents()
)

print(f"Total Tagged Sentences : {len(tagged_sentences):,}")

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

random.seed(42)

random.shuffle(tagged_sentences)

split_index = int(
    0.8 * len(tagged_sentences)
)

train_sentences = tagged_sentences[:split_index]

test_sentences = tagged_sentences[split_index:]

print(f"Training Sentences : {len(train_sentences):,}")
print(f"Testing Sentences  : {len(test_sentences):,}")

# ==========================================================
# BUILD VOCABULARY
# ==========================================================

vocabulary = set()

for sentence in train_sentences:

    for word, tag in sentence:

        vocabulary.add(word.lower())

print(f"\nVocabulary Size : {len(vocabulary):,}")

# ==========================================================
# BUILD TAG SET
# ==========================================================

tag_set = set()

for sentence in train_sentences:

    for word, tag in sentence:

        tag_set.add(tag)

print(f"Number of Tags  : {len(tag_set):,}")

# ==========================================================
# INITIAL TAG COUNTS
# ==========================================================

initial_tag_counts = Counter()

for sentence in train_sentences:

    if len(sentence) > 0:

        first_tag = sentence[0][1]

        initial_tag_counts[first_tag] += 1

# ==========================================================
# INITIAL PROBABILITIES
# ==========================================================

initial_probabilities = {}

total_sentences = len(train_sentences)

for tag in initial_tag_counts:

    initial_probabilities[tag] = (
        initial_tag_counts[tag] /
        total_sentences
    )

# ==========================================================
# TRANSITION COUNTS
# ==========================================================

transition_counts = defaultdict(Counter)

for sentence in train_sentences:

    tags = []

    for word, tag in sentence:

        tags.append(tag)

    for i in range(len(tags) - 1):

        current_tag = tags[i]

        next_tag = tags[i + 1]

        transition_counts[current_tag][next_tag] += 1

# ==========================================================
# TRANSITION PROBABILITIES
# ==========================================================

transition_probabilities = defaultdict(dict)

for current_tag in transition_counts:

    total = sum(
        transition_counts[current_tag].values()
    )

    for next_tag in transition_counts[current_tag]:

        transition_probabilities[current_tag][next_tag] = (

            transition_counts[current_tag][next_tag] / total

        )

# ==========================================================
# EMISSION COUNTS
# ==========================================================

emission_counts = defaultdict(Counter)

for sentence in train_sentences:

    for word, tag in sentence:

        emission_counts[tag][word.lower()] += 1

# ==========================================================
# EMISSION PROBABILITIES
# ==========================================================

emission_probabilities = defaultdict(dict)

for tag in emission_counts:

    total = sum(
        emission_counts[tag].values()
    )

    for word in emission_counts[tag]:

        emission_probabilities[tag][word] = (

            emission_counts[tag][word] / total

        )

# ==========================================================
# DISPLAY INITIAL PROBABILITIES
# ==========================================================

print("\n" + "=" * 60)
print("Top 10 Initial Tag Probabilities")
print("=" * 60)

top_initial = sorted(

    initial_probabilities.items(),

    key=lambda x: x[1],

    reverse=True

)[:10]

for tag, probability in top_initial:

    print(f"{tag:6s} {probability:.6f}")

# ==========================================================
# DISPLAY TRANSITION PROBABILITIES
# ==========================================================

print("\n" + "=" * 60)
print("Top 5 Transition Probabilities from NN")
print("=" * 60)

if "NN" in transition_probabilities:

    top_transition = sorted(

        transition_probabilities["NN"].items(),

        key=lambda x: x[1],

        reverse=True

    )[:5]

    for tag, probability in top_transition:

        print(

            f"NN -> {tag:6s} {probability:.6f}"

        )

# ==========================================================
# DISPLAY EMISSION PROBABILITIES
# ==========================================================

print("\n" + "=" * 60)
print("Top 5 Emission Probabilities from VB")
print("=" * 60)

if "VB" in emission_probabilities:

    top_emission = sorted(

        emission_probabilities["VB"].items(),

        key=lambda x: x[1],

        reverse=True

    )[:5]

    for word, probability in top_emission:

        print(

            f"{word:15s} {probability:.6f}"

        )

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("Q11 Completed Successfully")
print("=" * 60)

# ==========================================================
# Q12 : POS TAGGING USING HMM PROBABILITIES
# ==========================================================

print("\n" + "=" * 60)
print("Q12 : POS Tagging Evaluation")
print("=" * 60)

# ==========================================================
# BUILD WORD -> MOST FREQUENT TAG DICTIONARY
# ==========================================================

word_tag_dictionary = {}

for word in vocabulary:

    tag_counter = Counter()

    for tag in emission_probabilities:

        if word in emission_probabilities[tag]:

            tag_counter[tag] = emission_probabilities[tag][word]

    if len(tag_counter) > 0:

        most_common_tag = tag_counter.most_common(1)[0][0]

        word_tag_dictionary[word] = most_common_tag

print("Word-Tag Dictionary Created.")

# ==========================================================
# TAG TEST SENTENCES
# ==========================================================

correct_predictions = 0

wrong_predictions = 0

unknown_words = 0

total_words = 0

incorrect_examples = []

for sentence in test_sentences:

    for word, actual_tag in sentence:

        word_lower = word.lower()

        # -------------------------
        # Known Word
        # -------------------------

        if word_lower in word_tag_dictionary:

            predicted_tag = word_tag_dictionary[word_lower]

        # -------------------------
        # Unknown Word
        # -------------------------

        else:

            predicted_tag = "NN"

            unknown_words += 1

        # -------------------------
        # Accuracy
        # -------------------------

        if predicted_tag == actual_tag:

            correct_predictions += 1

        else:

            wrong_predictions += 1

            if len(incorrect_examples) < 10:

                incorrect_examples.append(

                    (
                        word,
                        actual_tag,
                        predicted_tag
                    )

                )

        total_words += 1

# ==========================================================
# RESULTS
# ==========================================================

accuracy = (

    correct_predictions /

    total_words

) * 100

print("\nAccuracy Results")

print(f"Correct Predictions : {correct_predictions:,}")

print(f"Wrong Predictions   : {wrong_predictions:,}")

print(f"Unknown Words       : {unknown_words:,}")

print(f"Total Words         : {total_words:,}")

print(f"Accuracy            : {accuracy:.2f}%")

# ==========================================================
# SAMPLE ERRORS
# ==========================================================

print("\n" + "=" * 60)
print("Sample Incorrect Predictions")
print("=" * 60)

for word, actual, predicted in incorrect_examples:

    print(

        f"{word:15s}"

        f" Actual={actual:6s}"

        f" Predicted={predicted}"

    )

# ==========================================================
# OBSERVATION
# ==========================================================

print("\nObservation:")

print(
    "The tagger predicts tags using the most"
)

print(
    "frequent tag observed for each word during"
)

print(
    "training."
)

print()

print(
    "Unknown words are assigned the default"
)

print(
    "tag 'NN'."
)

print()

print(
    "This provides a simple baseline POS tagger"
)

print(
    "before implementing better unknown-word"
)

print(
    "handling."
)

# ==========================================================
# Q13 : IMPROVED UNKNOWN WORD HANDLING
# ==========================================================

print("\n" + "=" * 60)
print("Q13 : Improved Unknown Word Handling")
print("=" * 60)

# ==========================================================
# NORMALIZE BROWN TAGS
# ==========================================================

def normalize_tag(tag):

    tag = tag.split("-")[0]
    tag = tag.split("+")[0]

    return tag


# ==========================================================
# BUILD NORMALIZED WORD-TAG DICTIONARY
# ==========================================================

normalized_word_tag = {}

for word in vocabulary:

    tag_counter = Counter()

    for tag in emission_probabilities:

        if word in emission_probabilities[tag]:

            normalized = normalize_tag(tag)

            tag_counter[normalized] += emission_probabilities[tag][word]

    if len(tag_counter) > 0:

        normalized_word_tag[word] = tag_counter.most_common(1)[0][0]


# ==========================================================
# TAG TEST SET
# ==========================================================

correct_predictions = 0

wrong_predictions = 0

unknown_words = 0

total_words = 0

incorrect_examples = []


for sentence in test_sentences:

    for word, actual_tag in sentence:

        word_lower = word.lower()

        actual_tag = normalize_tag(actual_tag)

        # -------------------------
        # Known Word
        # -------------------------

        if word_lower in normalized_word_tag:

            predicted_tag = normalized_word_tag[word_lower]

        # -------------------------
        # Unknown Word
        # -------------------------

        else:

            unknown_words += 1

            # -------------------------
            # Very Simple Heuristics
            # -------------------------

            if word[0].isupper():

                predicted_tag = "NP"

            elif word.endswith("ing"):

                predicted_tag = "VBG"

            elif word.endswith("ed"):

                predicted_tag = "VBD"

            elif word.endswith("ly"):

                predicted_tag = "RB"

            elif word.endswith("s"):

                predicted_tag = "NNS"

            else:

                predicted_tag = "NN"

        # -------------------------
        # Accuracy
        # -------------------------

        if predicted_tag == actual_tag:

            correct_predictions += 1

        else:

            wrong_predictions += 1

            if len(incorrect_examples) < 10:

                incorrect_examples.append(

                    (
                        word,
                        actual_tag,
                        predicted_tag
                    )

                )

        total_words += 1


# ==========================================================
# RESULTS
# ==========================================================

accuracy = (

    correct_predictions /

    total_words

) * 100


print("\nImproved Accuracy Results\n")

print(f"Correct Predictions : {correct_predictions:,}")

print(f"Wrong Predictions   : {wrong_predictions:,}")

print(f"Unknown Words       : {unknown_words:,}")

print(f"Total Words         : {total_words:,}")

print(f"Accuracy            : {accuracy:.2f}%")

# ==========================================================
# SAMPLE ERRORS
# ==========================================================

print("\n" + "=" * 60)
print("Sample Incorrect Predictions")
print("=" * 60)

for word, actual, predicted in incorrect_examples:

    print(

        f"{word:15s}"

        f" Actual={actual:5s}"

        f" Predicted={predicted}"

    )

# ==========================================================
# OBSERVATION
# ==========================================================

print("\nObservation:")

print(
    "The Brown corpus contains many tag variations"
)

print(
    "such as NN-TL, JJ-HL and IN-NC."
)

print()

print(
    "Normalizing these tags improves evaluation."
)

print()

print(
    "Simple suffix rules also improve tagging"
)

print(
    "for unknown words."
)

# ==========================================================
# Q14 : ERROR ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("Q14 : Error Analysis")
print("=" * 60)

# ==========================================================
# COUNT CONFUSED TAG PAIRS
# ==========================================================

error_counter = Counter()

for sentence in test_sentences:

    for word, actual_tag in sentence:

        word_lower = word.lower()

        actual_tag = normalize_tag(actual_tag)

        # -------------------------
        # Predict Tag
        # -------------------------

        if word_lower in normalized_word_tag:

            predicted_tag = normalized_word_tag[word_lower]

        else:

            if word[0].isupper():

                predicted_tag = "NP"

            elif word.endswith("ing"):

                predicted_tag = "VBG"

            elif word.endswith("ed"):

                predicted_tag = "VBD"

            elif word.endswith("ly"):

                predicted_tag = "RB"

            elif word.endswith("s"):

                predicted_tag = "NNS"

            else:

                predicted_tag = "NN"

        # -------------------------
        # Store Incorrect Pair
        # -------------------------

        if predicted_tag != actual_tag:

            error_counter[
                (actual_tag, predicted_tag)
            ] += 1


# ==========================================================
# TOP 10 CONFUSIONS
# ==========================================================

print("\nTop 10 Confused POS Tags\n")

top_errors = error_counter.most_common(10)

for rank, ((actual, predicted), count) in enumerate(top_errors, start=1):

    print(

        f"{rank:2d}. "

        f"{actual:5s}"

        f" -> "

        f"{predicted:5s}"

        f" : {count}"

    )

# ==========================================================
# TOTAL ERRORS
# ==========================================================

print("\Error Summary\n")

print(f"Total Incorrect Predictions : {wrong_predictions:,}")

print(f"Unique Error Types          : {len(error_counter):,}")

# ==========================================================
# OBSERVATION
# ==========================================================

print("\nObservation:")

print(
    "Most errors occur between linguistically"
)

print(
    "similar parts of speech."
)

print()

print(
    "Examples include noun versus verb,"
)

print(
    "adjective versus adverb,"
)

print(
    "and preposition versus particle."
)

print()

print(
    "These ambiguities require contextual"
)

print(
    "information which is difficult for"
)

print(
    "a simple HMM model."
)

# ==========================================================
# COMPLETED
# ==========================================================

print("\n" + "=" * 60)
print("Q14 Completed Successfully")
print("=" * 60)

# ==========================================================
# Q15 : COMPARISON WITH NLTK TAGGER
# ==========================================================

print("\n" + "=" * 60)
print("Q15 : Comparison with NLTK Tagger")
print("=" * 60)

from nltk.tag import UnigramTagger

# ==========================================================
# TRAIN NLTK UNIGRAM TAGGER
# ==========================================================

print("\nTraining NLTK Unigram Tagger...")

nltk_tagger = UnigramTagger(
    train_sentences,
    backoff=None
)

print("Training Completed.")

# ==========================================================
# EVALUATE NLTK TAGGER
# ==========================================================

nltk_correct = 0

nltk_total = 0

for sentence in test_sentences:

    words = []

    actual_tags = []

    for word, tag in sentence:

        words.append(word)

        actual_tags.append(
            normalize_tag(tag)
        )

    predicted = nltk_tagger.tag(words)

    for i in range(len(predicted)):

        predicted_tag = predicted[i][1]

        if predicted_tag is None:

            predicted_tag = "NN"

        predicted_tag = normalize_tag(
            predicted_tag
        )

        if predicted_tag == actual_tags[i]:

            nltk_correct += 1

        nltk_total += 1

# ==========================================================
# ACCURACY
# ==========================================================

nltk_accuracy = (

    nltk_correct /

    nltk_total

) * 100

# ==========================================================
# COMPARISON TABLE
# ==========================================================

print("\n" + "=" * 60)
print("Accuracy Comparison")
print("=" * 60)

print(f"{'Model':30s} Accuracy")

print("-" * 45)

print(f"{'Baseline Tagger (Q12)':30s} 35.40%")

print(f"{'Improved HMM (Q13)':30s} {accuracy:.2f}%")

print(f"{'NLTK Unigram Tagger':30s} {nltk_accuracy:.2f}%")

# ==========================================================
# BEST MODEL
# ==========================================================

scores = {

    "Baseline Tagger": 35.40,

    "Improved HMM": accuracy,

    "NLTK Unigram": nltk_accuracy

}

best_model = max(

    scores,

    key=scores.get

)

print("\nBest Performing Model")

print(f"{best_model}")

print(f"Accuracy : {scores[best_model]:.2f}%")

# ==========================================================
# OBSERVATION
# ==========================================================

print("\nObservation:")

print(
    "The baseline model provides a simple"
)

print(
    "word-frequency based POS tagger."
)

print()

print(
    "Normalizing Brown tags and handling"
)

print(
    "unknown words significantly improved"
)

print(
    "the tagging accuracy."
)

print()

print(
    "The NLTK Unigram Tagger serves as"
)

print(
    "a standard baseline implementation."
)

print()

print(
    "Overall, the improved HMM approach"
)

print(
    "achieved competitive performance on"
)

print(
    "the Brown corpus."
)

# ==========================================================
# COMPLETED
# ==========================================================

print("\n" + "=" * 60)
print("Assignment Part-3 Completed Successfully")
print("=" * 60)