#====================================================================
# Import Required Libraries
#====================================================================

import os
import string
from collections import Counter
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

#====================================================================
# Download Required Datasets
#====================================================================

nltk.download("brown")
nltk.download("punkt")

#====================================================================
# Create Output Directory
#====================================================================

OUTPUT_FOLDER = "output"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

#====================================================================
# Load Brown Corpus
#====================================================================

from nltk.corpus import brown

print("=" * 60)
print("Loading Brown Corpus...")
print("=" * 60)

tokens = brown.words()

print(f"Total Tokens : {len(tokens):,}")

#====================================================================
# Text Preprocessing
#====================================================================
processed_tokens = []

for token in tokens:
    token = token.lower()
    token = token.strip(string.punctuation)

    if token != "":
        processed_tokens.append(token)

print(f"Processed Tokens:  {len(processed_tokens):,}")

#====================================================================
# Count Word Frequency
#====================================================================

word_counter = Counter(processed_tokens)

print(f"Vocabulary Size: {len(word_counter):,}")

#====================================================================
# Rank Words : Sort words by frequency
#====================================================================

sorted_words = word_counter.most_common()


#====================================================================
# Top 10 Words
#====================================================================

top10_words = pd.DataFrame(
    sorted_words[:10],
    columns=[
        "Word",
        "Frequency"
    ]
)

top10_words.index = top10_words.index + 1
top10_words.index.name = "Rank"

print("\n Top 10 most frequent words\n")
print(top10_words)

#====================================================================
# Save the table
#====================================================================

output_file = os.path.join(
    OUTPUT_FOLDER,
    "q1_top10_words.csv"
)

top10_words.to_csv(
    output_file,
    index=True
)

print("\n Top 10 table saved successfully\n")

#====================================================================
# Q2 -  Prepare Rank and Frequency
#====================================================================

print("\n" + "=" * 60)
print("Zipf's Law Analysis")
print("="*60)

frequencies = []

for word, frequency in sorted_words:
    frequencies.append(frequency)

ranks = list(range(1, len(frequencies)+1))

#====================================================================
# Convert to Log Scale
#====================================================================
log_rank = np.log(ranks)
log_frequency = np.log(frequencies)

#====================================================================
# Linear Regression
#====================================================================
slope, intercept, r_value, p_value, std_error = linregress(
    log_rank,
    log_frequency
)

print(f"Slope      : {slope:.4f}")
print(f"Intercept  : {intercept:.4f}")
print(f"R-Squared  : {r_value ** 2:.4f}")

#====================================================================
# Plot Zipf's Curve
#====================================================================
plt.figure(figsize=(8, 6))

plt.scatter(
    log_rank,
    log_frequency,
    s=8,
    alpha=0.5,
    label="Word Frequencies"
)

plt.plot(
    log_rank,
    intercept + slope * log_rank,
    color="red",
    linewidth=2,
    label="Linear Regression"
)

plt.title("Zipf's Law (Brown Corpus)")
plt.xlabel("log(Rank)")
plt.ylabel("log(Frequency)")
plt.legend()
plt.grid(True)

#====================================================================
# Save Figure
#====================================================================

plot_file = os.path.join(
    OUTPUT_FOLDER,
    "q2_zipf_plot.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
print("\nZipf plot saved successfully.")

#====================================================================
# Observation
#====================================================================
print("\nObservation:")

if -1.4 <= slope <= -0.8:
    print("The slope is close to -1.")
    print("The Brown corpus approximately follows Zipf's Law.")
else:
    print("The slope deviates from -1.")
    print("The Brown corpus does not perfectly follow Zipf's Law.")

#====================================================================
# Q3 - Top 20 and Bottom-20 words
#====================================================================

print("\n" + "=" * 60)
print("Q3 : Top-20 and Bottom-20 Words")
print("=" * 60)

top20_data_frame = pd.DataFrame(
    sorted_words[:20],
    columns=["Word","Frequency"]
)

top20_data_frame.index = top20_data_frame.index+1
top20_data_frame.index.name = "Rank"

print("\nTop 20 Most Frequent Words\n")

print(top20_data_frame)

bottom20_data_frame = pd.DataFrame(
    sorted_words[-20:],
    columns=["Word","Frequency"]
)

bottom20_data_frame.index = bottom20_data_frame.index+1
bottom20_data_frame.index.name = "Rank"

print("\nBottom 20 Most Frequent Words\n")

print(bottom20_data_frame)

#====================================================================
# Save Tables
#====================================================================

top20_data_frame.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "q3_top20_words.csv"
    ),
    index=True
)

bottom20_data_frame.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "q3_bottom20_words.csv"
    ),
    index=True
)

#====================================================================
# Observation
#====================================================================

print("\nObservation:")

print("- The Top-20 words are mainly closed-class words, such as articles (the, a), prepositions (of, in, on), conjunctions (and), pronouns (he, his, it), and auxiliary verbs (is, was, be).")
print("- These grammatical words occur frequently because they are essential for constructing English sentences.")

print()

print("- In contrast, the Bottom-20 words are mostly open-class words,")
print("  including nouns, adjectives, and domain-specific vocabulary such as bodhisattva, bilharziasis, and stupefying.")
print("- These words occur only once in the corpus because open-class vocabulary is much larger and continually expands.")

print()

print("  This demonstrates the type-token distinction: a small set of closed-class words contributes a large proportion of the corpus tokens,")
print(" while a large number of open-class words contribute to vocabulary diversity.")

#====================================================================
# Q4 - Type Token Ratio (TTR)
#====================================================================

print("\n" + "=" * 60)
print("Q4 : Type Token Ratio")
print("=" * 60)

corpus_sizes = [
    10000,
    50000,
    100000,
    500000,
    1000000
]

#====================================================================
# Compute TTR
#====================================================================

ttr_values = []

for size in corpus_sizes:
    current_tokens = processed_tokens[:size]

    vocabulary = set(current_tokens)

    ttr = len(vocabulary) / len(current_tokens)

    ttr_values.append(ttr)

    print(f"Corpus Size : {size:,}")

    print(f"Vocabulary  : {len(vocabulary):,}")

    print(f"TTR : {ttr:.4f}\n")
    

#====================================================================
# Save Table
#====================================================================

ttr_df = pd.DataFrame({
    "Corpus Size": corpus_sizes,
    "TTR": ttr_values
})

ttr_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "q4_ttr.csv"
    ),

    index=False
)

print("TTR table saved successfully.")

#====================================================================
# Plot
#====================================================================
plt.figure(figsize=(8,5))

plt.plot(
    corpus_sizes,
    ttr_values,
    marker="o",
    linewidth=2,
    markersize=6
)

plt.xticks(
    corpus_sizes,
    [
        "10K",
        "50K",
        "100K",
        "500K",
        "1M"
    ]
)
plt.xticks(
    corpus_sizes,
    [
        "10K",
        "50K",
        "100K",
        "500K",
        "1M"
    ]
)

plt.title("Type Token Ratio vs Corpus Size")
plt.xlabel("Corpus Size")
plt.ylabel("TTR")
plt.grid(True)
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "q4_ttr_plot.png"
    ),
        dpi=300,
        bbox_inches="tight"
)

plt.show()
print("TTR plot saved successfully.")

#====================================================================
# Observation
#====================================================================

print("\nObservation:")

print("The Type-Token Ratio (TTR) decreases as the corpus size increases.")
print("At the beginning, many new words are encountered, resulting in a high TTR.")
print("As more text is processed, previously seen words appear repeatedly while the number of new words grows more slowly.")
print("Therefore, vocabulary growth is slower than token growth, causing the TTR to decrease with increasing corpus size.")

#====================================================================
# Q5 - MATTR
#====================================================================
print("\n" + "=" * 60)
print("Q5 : Moving Average Type Token Ratio")
print("=" * 60)


def compute_mattr(tokens, window_size=1000):

    if len(tokens) < window_size:
        return 0

    ttr_values = []

    for i in range(len(tokens) - window_size + 1):

        window = tokens[i:i + window_size]

        vocabulary = set(window)

        ttr = len(vocabulary) / window_size

        ttr_values.append(ttr)

    return sum(ttr_values) / len(ttr_values)

#====================================================================
# Compute MATTR
#====================================================================

mattr_values = []

for size in corpus_sizes:

    current_tokens = processed_tokens[:size]

    mattr = compute_mattr(
        current_tokens,
        window_size=1000
    )

    mattr_values.append(mattr)

    print(f"Corpus Size : {size:,}")

    print(f"MATTR       : {mattr:.4f}\n")

#====================================================================
# Save MATTR Table
#====================================================================
mattr_df = pd.DataFrame({

    "Corpus Size": corpus_sizes,

    "MATTR": mattr_values

})

mattr_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "q5_mattr.csv"
    ),

    index=False
)

print("MATTR table saved successfully.")

#====================================================================
# MATTR Plot
#====================================================================
plt.figure(figsize=(8,5))

plt.plot(
    corpus_sizes,
    mattr_values,
    marker="o",
    linewidth=2,
    markersize=6,
    color="green"
)

plt.xticks(
    corpus_sizes,
    [
        "10K",
        "50K",
        "100K",
        "500K",
        "1M"
    ]
)

plt.xticks(
    corpus_sizes,
    [
        "10K",
        "50K",
        "100K",
        "500K",
        "1M"
    ]
)

plt.title("Moving Average TTR vs Corpus Size")
plt.xlabel("Corpus Size")
plt.ylabel("MATTR")
plt.grid(True)
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "q5_mattr_plot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("MATTR plot saved successfully.")

#====================================================================
# Observation
#====================================================================

print("\nObservation:")

print("Unlike the basic Type-Token Ratio (TTR), the Moving Average Type-Token Ratio (MATTR) changes only slightly as the corpus size increases.")
print("Since MATTR computes lexical diversity over fixed-size moving windows,")
print(" it is less affected by corpus size and provides a more stable estimate of lexical diversity. Therefore, MATTR is more suitable for comparing corpora of different sizes.")