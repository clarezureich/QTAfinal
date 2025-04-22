#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Collection 
Part 1
Scraping Guiness Archive and IFI websites for Old Advert Descriptions 
"""
#--------------Import Packages--------------#
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import nltk
#nltk.download('punkt_tab')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger_eng')
import seaborn as sns 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from afinn import Afinn
from nrclex import NRCLex
from empath import Empath
from collections import defaultdict


#--------------IFI Scraping--------------#
BASE_URL = "https://ifiarchiveplayer.ie/guinness-ads/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_ad_links():
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    
    links = soup.find_all("a", class_="vc_gitem-link vc-zone-link")
    urls = []
    for link in links:
        href = link.get("href")
        if href and href.startswith("https://ifiarchiveplayer.ie/"):
            urls.append(href)
    return list(set(urls))  # remove duplicates

def extract_metadata(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    def get_nth_metainfo(n):
        elements = soup.select("div.archive-metainfo")
        if len(elements) >= n:
            return elements[n - 1].get_text(strip=True)
        return None

    def get_text(selector):
        el = soup.select_one(selector)
        return el.get_text(strip=True) if el else None

    return {
        "URL": url,
        "Title": get_text("h1.entry-title"),
        "Category": get_nth_metainfo(1),
        "Director": get_nth_metainfo(2),
        "Producer": get_nth_metainfo(3),
        "Year": get_nth_metainfo(4),
        "Duration": get_nth_metainfo(5),
        "Language": get_nth_metainfo(6),
        "Description": get_text("span.NormalTextRun")
    }


ad_links = get_ad_links()
print(f"Found {len(ad_links)} ad links.")

all_data = []
for i, link in enumerate(ad_links):
    print(f"[{i+1}/{len(ad_links)}] Scraping: {link}")
    data = extract_metadata(link)
    if data:
        all_data.append(data)
    time.sleep(1.5)  # respectful delay

# Save results to CSV
ifi_metadata = pd.DataFrame(all_data)
ifi_metadata.to_csv("ifi_archive_player.csv", index=False)
print("Metadata saved to 'ifi_archive_player.csv'")




#--------------Guinness Archive Scraping--------------#
#Collect all video urls from the Guinness Archive Website

def collect_video_urls(max_scrolls=25, sleep_between_scrolls=3):
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options)
    
    url = "https://guinness.access.preservica.com/?s=video"
    driver.get(url)
    time.sleep(5)

    video_urls = set()
    scrolls = 0

    print("Starting scroll and URL collection...")

    while scrolls < max_scrolls:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_scrolls)

        # Use the selector
        a_elements = driver.find_elements(By.CSS_SELECTOR, 'div.archive_name h5 a')
        for a in a_elements:
            href = a.get_attribute("href")
            if href:
                video_urls.add(href)

        scrolls += 1
        print(f"Scroll {scrolls}/{max_scrolls}: {len(video_urls)} total unique URLs")

    # Save to CSV
    guinness_urls = pd.DataFrame(video_urls, columns=["URL"])
    guinness_urls.to_csv("guinness_video_urls.csv", index=False)
    print(f"\n Done, saved {len(guinness_urls)} URLs to 'guinness_video_urls.csv'")

    driver.quit()

# Run the collector
if __name__ == "__main__":
    collect_video_urls()
    

#Scrape metadata from Guinness archive using all URLs collected above
def scrape_guinness_video_metadata(input_csv="guinness_video_urls.csv", output_csv="guinness_video_archive.csv"):
    df_urls = pd.read_csv(input_csv)
    urls = df_urls["URL"].dropna().unique().tolist()

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options)

    data = []

    for i, video_url in enumerate(urls):
        print(f"Scraping ({i+1}/{len(urls)}): {video_url}")
        driver.get(video_url)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="page_wrapper"]/div/div/section/aside'))
            )
        except TimeoutException:
            print("Timeout while waiting for content to load.")
            continue

        try:
            title = driver.find_element(By.XPATH, '//*[@id="page_wrapper"]/div/div/section/aside/div/div[1]/h5').text.strip()
        except NoSuchElementException:
            title = ""

        try:
            description = driver.find_element(By.XPATH, '//*[@id="page_wrapper"]/div/div/section/aside/div/div[1]/p').text.strip()
        except NoSuchElementException:
            description = ""

        try:
            date = driver.find_element(By.XPATH, '//*[@id="page_wrapper"]/div/div/section/aside/div/div[2]/p[2]/span[3]').text.strip()
        except NoSuchElementException:
            date = ""

        data.append({
            "URL": video_url,
            "Title": title,
            "Description": description,
            "Year": date
        })

    driver.quit()

    guinness_archive_metadata = pd.DataFrame(data)
    guinness_archive_metadata.to_csv(output_csv, index=False)
    print(f"\nDONE, saved metadata for {len(guinness_archive_metadata)} videos to '{output_csv}'")

# Run it!
if __name__ == "__main__":
    scrape_guinness_video_metadata()



"""
Data Analysis 
Part 2
QTA Text Methods Applied to "Archived Guinness Adverts"
"""
#Load in two old-time advertisement datasets
#Scrapped from "guinness_ads code" 

# Load datasets with fallback encoding
ifi_metadata = pd.read_csv("/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Data Sets/ifi_archive_player.csv", encoding="ISO-8859-1")


guinness_archive_metadata = pd.read_csv("/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Data Sets/guinness_video_archive.csv",encoding="ISO-8859-1")

# Standardize column names for easier comparison
ifi_metadata.columns = ifi_metadata.columns.str.strip().str.lower()
guinness_archive_metadata.columns = guinness_archive_metadata.columns.str.strip().str.lower()

# Clean and compare titles
guinness_titles = set(guinness_archive_metadata['title'].str.strip().str.lower())
ifi_metadata['title_clean'] = ifi_metadata['title'].str.strip().str.lower()
ifi_unique = ifi_metadata[~ifi_metadata['title_clean'].isin(guinness_titles)].copy()

# Drop helper column
ifi_unique.drop(columns=['title_clean', 'category', 'director', 'producer', 'duration' ,'language'], inplace=True)

# Merge unique IFI ads into Guinness archive metadata
merged_metadata = pd.concat([guinness_archive_metadata, ifi_unique], ignore_index=True)

# Drop rows where description is NaN or only whitespace
merged_metadata = merged_metadata.dropna(subset=['description'])

merged_metadata.head()

#Preprocessing and cleaning text 

# Remove blank or missing descriptions
merged_metadata = merged_metadata[merged_metadata['description'].str.strip() != '']
merged_metadata = merged_metadata.dropna(subset=['description'])

# Define stopwords
stop_words = set(stopwords.words('english'))

# Filter out production related words - not related to sentiment 
extra_stopwords = {
    "director", "producer", "agency", "company", "jwt", "advertisement",
    "commercials", "clio", "award", "awards", "international", "festival", 
    "production", "presents", "ltd", "television", "colour", "black", "white"
}

#Combine stop word lists 
stop_words = stop_words.union(extra_stopwords)

# Function to clean + tokenize
def clean_tokenize(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation + numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Remove encoded characters from descriptions
def clean_description(text):
    # Strip typical misencoded characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()
# Apply to merged dataset
merged_metadata['description'] = merged_metadata['description'].apply(clean_description)

# Clean the year column: remove 's', drop 'nan', strip decimals, convert to int
merged_metadata['year'] = (
    merged_metadata['year']
    .astype(str)
    .str.replace("s", "")
    .str.strip()
    .str.replace(".0", "", regex=False)  # remove .0 from '1970.0'
)

merged_metadata['year'] = merged_metadata['year'].astype(int)

# Apply to descriptions
merged_metadata['tokens'] = merged_metadata['description'].apply(clean_tokenize)

# Word Cloud and Frequency Plot

# Flatten all tokens into a single list
all_tokens = [token for tokens in merged_metadata['tokens'] for token in tokens]

# Count word frequencies
word_counts = Counter(all_tokens)

# Top 20 words bar plot
top_words = word_counts.most_common(20)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 20 Most Common Words in Guinness Ad Descriptions")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Guinness Ad Descriptions", fontsize=16)
plt.show()

# VADER Sentiment Analysis
vader = SentimentIntensityAnalyzer()

# Apply VADER to get the compound sentiment score for each ad
merged_metadata['vader_score'] = merged_metadata['description'].apply(
    lambda x: vader.polarity_scores(x)['compound']
)

# Basic sentiment label
def label_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

merged_metadata['vader_label'] = merged_metadata['vader_score'].apply(label_sentiment)

# Preview
print(merged_metadata[['title', 'year', 'vader_score', 'vader_label', 'description']].head())

#AFINN Sentiment Analysis to each description -> Compare to Vader
afinn = Afinn()
merged_metadata['afinn_score'] = merged_metadata['description'].apply(lambda x: afinn.score(x))

# VADER Sentiment Analysis
vader = SentimentIntensityAnalyzer()

# Apply VADER to get the compound sentiment score for each ad
merged_metadata['vader_score'] = merged_metadata['description'].apply(
    lambda x: vader.polarity_scores(x)['compound']
)

# Basic sentiment label
def label_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

merged_metadata['vader_label'] = merged_metadata['vader_score'].apply(label_sentiment)

# Preview
print(merged_metadata[['title', 'year', 'vader_score', 'vader_label', 'description']].head())

#AFINN Sentiment Analysis to each description -> Compare to Vader
afinn = Afinn()
merged_metadata['afinn_score'] = merged_metadata['description'].apply(lambda x: afinn.score(x))

# LDA Topic Modeling 

# Convert tokens to strings (for CountVectorizer)
merged_metadata['cleaned_text'] = merged_metadata['tokens'].apply(lambda x: ' '.join(x))

# Create DTM (Document-Term Matrix)
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
dtm = vectorizer.fit_transform(merged_metadata['cleaned_text'])

# Fit LDA model
lda = LatentDirichletAllocation(n_components=4, random_state=42)  
lda.fit(dtm)

# Get top words per topic
def print_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx + 1}")
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print("  " + ", ".join(top_features))

feature_names = vectorizer.get_feature_names_out()
print_topics(lda, feature_names)


# Topic distribution added to each Ad 
topic_values = lda.transform(dtm)
merged_metadata['topic'] = topic_values.argmax(axis=1)

# Sample ads descriptions from each topic 
for topic_num in range(4):
    print(f"\n Sample Ads from Topic {topic_num + 1}")
    topic_ads = merged_metadata[merged_metadata['topic'] == topic_num]
    if not topic_ads.empty:
        print(topic_ads[['title', 'description']].head(3))
    else:
        print("  (No ads in this topic!)")


# Number of ads in each topic 
merged_metadata['topic'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Ads by Topic")
plt.xlabel("Topic")
plt.ylabel("Number of Ads")
plt.xticks(ticks=range(5), labels=[f"Topic {i+1}" for i in range(5)], rotation=0)
plt.tight_layout()
plt.show()

# Topic vs. Sentiment 
sns.boxplot(x='topic', y='vader_score', data=merged_metadata)
plt.title("Sentiment Distribution by Topic")
plt.xlabel("Topic")
plt.ylabel("VADER Sentiment Score")
plt.show()

# Topic trend over time 
merged_metadata['year'] = merged_metadata['year'].astype(str).str[:4]
topic_by_year = merged_metadata.groupby(['year', 'topic']).size().unstack(fill_value=0)

topic_by_year.plot(kind='line', figsize=(12,6))
plt.title("Topic Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Ads")
plt.legend(title="Topic")
plt.tight_layout()
plt.show()

# KWIC for gendered words 
def kwic(text, keywords, window=5):
    tokens = text.lower().split()
    results = []
    for i, token in enumerate(tokens):
        if token in keywords:
            left = " ".join(tokens[max(i - window, 0):i])
            right = " ".join(tokens[i + 1:i + 1 + window])
            results.append(f"... {left} >>{token}<< {right} ...")
    return results

# Define keyword sets
female_terms = {"woman", "women", "lady", "female", "girl"}
male_terms = {"man", "men", "male", "barman", "gentleman", "lad"}

# Run for each row
#print("\nKWIC for Woman/Women:\n")
#for idx, row in merged_metadata.iterrows():
#    matches = kwic(row['description'], keywords=female_terms, window=5)
#    for match in matches:
#        print(f"[{row['title']}] {match}")

#print("\nKWIC for Man/Men:\n")
#for idx, row in merged_metadata.iterrows():
#    matches = kwic(row['description'], keywords=male_terms, window=5)
#    for match in matches:
#        print(f"[{row['title']}] {match}")


# Dictionary Analysis 
###NRC###
def get_nrc_scores(text):
    emotions = NRCLex(text).raw_emotion_scores
    return emotions

# Apply NRC to each description
merged_metadata['nrc_emotions'] = merged_metadata['description'].apply(get_nrc_scores)

# Explode to columns
nrc_df = merged_metadata['nrc_emotions'].apply(pd.Series).fillna(0)
merged_metadata = pd.concat([merged_metadata, nrc_df], axis=1)

###Empath###
lexicon = Empath()

# Select categories to score
categories = ["masculinity", "feminine", "alcohol", "affection", "trust", "anger"]

def get_empath_scores(text):
    return lexicon.analyze(text, categories=categories, normalize=True)

merged_metadata['empath'] = merged_metadata['description'].apply(get_empath_scores)

# Expand to columns
empath_df = merged_metadata['empath'].apply(pd.Series).fillna(0)
merged_metadata = pd.concat([merged_metadata, empath_df], axis=1)

# Temporal Trend Analysis

# Convert year to integer 
merged_metadata['year'] = merged_metadata['year'].astype(int)

# Group years by decade
merged_metadata['decade'] = (merged_metadata['year'] // 10) * 10

# Group by decade and average sentiment scores
sentiment_by_decade = (
    merged_metadata
    .groupby('decade')[['vader_score', 'afinn_score']]
    .mean()
    .reset_index()
)

# Sort decades
sentiment_by_decade = sentiment_by_decade.sort_values('decade')

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=sentiment_by_decade, x='decade', y='vader_score', label='VADER', marker='o')
sns.lineplot(data=sentiment_by_decade, x='decade', y='afinn_score', label='AFINN', marker='s')

plt.title('Sentiment Trend in Guinness Ads by Decade', fontsize=14)
plt.xlabel('Decade')
plt.ylabel('Average Sentiment Score')
plt.xticks(sentiment_by_decade['decade'], rotation=45)
plt.grid(True)
plt.legend(title='Sentiment Method')
plt.tight_layout()
plt.show()




#Tagging Gender Mentions 
def mentions_only_male(tokens):
    return any(w in tokens for w in male_terms) and not any(w in tokens for w in female_terms)

def mentions_only_female(tokens):
    return any(w in tokens for w in female_terms) and not any(w in tokens for w in male_terms)

def mentions_both(tokens):
    return any(w in tokens for w in male_terms) and any(w in tokens for w in female_terms)

# Apply to data
merged_metadata['mentions_only_male'] = merged_metadata['tokens'].apply(mentions_only_male)
merged_metadata['mentions_only_female'] = merged_metadata['tokens'].apply(mentions_only_female)
merged_metadata['mentions_both'] = merged_metadata['tokens'].apply(mentions_both)




#Compare Sentiment Scores 
# Average sentiment by gender group
sentiment_compare = merged_metadata.groupby(
    ['mentions_only_male', 'mentions_only_female', 'mentions_both']
)[['vader_score', 'afinn_score']].mean().reset_index()

print(sentiment_compare)

#Plot 
# Add group label
def label_gender_group(row):
    if row['mentions_only_male']:
        return "Only Male"
    elif row['mentions_only_female']:
        return "Only Female"
    elif row['mentions_both']:
        return "Both"
    else:
        return "None"

merged_metadata['gender_group'] = merged_metadata.apply(label_gender_group, axis=1)

sns.boxplot(data=merged_metadata, x='gender_group', y='vader_score')
plt.title("Sentiment by Gender Mentions in Guinness Ads")
plt.xlabel("Gender Group")
plt.ylabel("VADER Sentiment Score")
plt.tight_layout()
plt.show()

# Topics by Gender Group 
# Count topic frequency by group
topic_by_gender = merged_metadata.groupby(['gender_group', 'topic']).size().unstack().fillna(0)

# Normalize by group size (optional)
topic_by_gender_norm = topic_by_gender.div(topic_by_gender.sum(axis=1), axis=0)

# Plot
topic_by_gender_norm.T.plot(kind='bar', figsize=(12, 6))
plt.title("Topic Distribution by Gender Mentions")
plt.ylabel("Proportion of Ads")
plt.xlabel("LDA Topic")
plt.tight_layout()
plt.show()

#Extract verbs from KWIC window 
def kwic_context_multi(text, keywords, window=5):
    tokens = text.lower().split()
    results = []
    for i, token in enumerate(tokens):
        if token in keywords:
            left = tokens[max(i - window, 0):i]
            right = tokens[i + 1:i + 1 + window]
            context = left + [f"KWIC_{token}"] + right
            results.append(" ".join(context))
    return results


#Apply to female_terms and male_terms above
merged_metadata['kwic_female'] = merged_metadata['cleaned_text'].apply(lambda x: kwic_context_multi(x, female_terms))
merged_metadata['kwic_male'] = merged_metadata['cleaned_text'].apply(lambda x: kwic_context_multi(x, male_terms))

#Inspect results (flatten into list)
female_kwic_all = [item for sublist in merged_metadata['kwic_female'] for item in sublist]
male_kwic_all = [item for sublist in merged_metadata['kwic_male'] for item in sublist]

for context in female_kwic_all[:10]:
    print(context)
for context in male_kwic_all[:10]:
    print(context)
    
#Tally verbs that appear in KWIC windows 
#Function to extract verbs from KWIC lists: 
def extract_verbs(kwic_list):
    verbs = []
    for context in kwic_list:
        tokens = word_tokenize(context)
        tagged = pos_tag(tokens)
        verbs += [word for word, tag in tagged if tag.startswith('VB') and word.lower() not in stopwords.words('english')]
    return verbs

#Apply to female/male KWIC and Extract verbs and remove 'opens' and 'concludes' 
#Descriptions generally start with "ad opens on..." and end with "ad concludes..." --> not actual action verbs
remove_verbs = {"opens", "concludes"}

female_verbs = [v for v in extract_verbs(female_kwic_all) if v.lower() not in remove_verbs]
male_verbs = [v for v in extract_verbs(male_kwic_all) if v.lower() not in remove_verbs]

#Count top verbs 
print("Top verbs for women:")
print(Counter(female_verbs).most_common(10))

print("\nTop verbs for men:")
print(Counter(male_verbs).most_common(10))

def plot_top_verbs(verb_counts, title):
    top_verbs = verb_counts.most_common(10)
    words, counts = zip(*top_verbs)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(title)
    plt.xlabel("Verb")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

plot_top_verbs(Counter(female_verbs), "Top Verbs Near 'Woman/Women'")
plot_top_verbs(Counter(male_verbs), "Top Verbs Near 'Man/Men'")

#Verbs by Decade (KWIC Context) 

# Create a container for decade-based verbs
female_verbs_by_decade = defaultdict(list)
male_verbs_by_decade = defaultdict(list)

# Loop through the ads
for _, row in merged_metadata.iterrows():
    decade = row['decade']
    # Get KWIC contexts
    for context in row['kwic_female']:
        tokens = word_tokenize(context)
        tagged = pos_tag(tokens, lang='eng')
        female_verbs_by_decade[decade] += [word for word, tag in tagged if tag.startswith('VB') and word.lower() not in {'opens', 'concludes'}]

    for context in row['kwic_male']:
        tokens = word_tokenize(context)
        tagged = pos_tag(tokens, lang='eng')
        male_verbs_by_decade[decade] += [word for word, tag in tagged if tag.startswith('VB') and word.lower() not in {'opens', 'concludes'}]

# Count and display top verbs per decade
for decade in sorted(female_verbs_by_decade.keys()):
    print(f"\nTop verbs near 'woman/women' in {decade}s:")
    print(Counter(female_verbs_by_decade[decade]).most_common(5))

for decade in sorted(male_verbs_by_decade.keys()):
    print(f"\nTop verbs near 'man/men' in {decade}s:")
    print(Counter(male_verbs_by_decade[decade]).most_common(5))
    
def plot_verbs_by_decade(verb_dict, gender_label):
    for decade in sorted(verb_dict.keys()):
        top_verbs = Counter(verb_dict[decade]).most_common(5)
        if not top_verbs:
            continue
        verbs, counts = zip(*top_verbs)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(verbs), y=list(counts))
        plt.title(f"Top Verbs Near '{gender_label}' in {decade}s")
        plt.xlabel("Verb")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_verbs_by_decade(female_verbs_by_decade, "woman/women")
plot_verbs_by_decade(male_verbs_by_decade, "man/men")

# Count topics by gender group
topic_gender_counts = merged_metadata.groupby(['gender_group', 'topic']).size().reset_index(name='count')

# Normalize within each gender group
topic_gender_pivot = topic_gender_counts.pivot(index='gender_group', columns='topic', values='count').fillna(0)
topic_gender_normalized = topic_gender_pivot.div(topic_gender_pivot.sum(axis=1), axis=0)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(topic_gender_normalized, cmap='Blues', annot=True, fmt=".2f")
plt.title("Normalized Topic Distribution by Gender Group")
plt.xlabel("Topic")
plt.ylabel("Gender Group")
plt.tight_layout()
plt.show()

#Adjective Analysis
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

#Remove KWIC tagging
doc = nlp(context.replace("KWIC_", ""))

#Exclude gendered adjectives
exclude_adjs = {"female", "male", "masculine", "feminine"}

def extract_adjectives(kwic_list):
    adjectives = []
    for context in kwic_list:
        context = context.replace("KWIC_", "")
        doc = nlp(context)
        adjectives += [
            token.text.lower() for token in doc 
            if token.pos_ == 'ADJ' and token.text.lower() not in exclude_adjs
        ]
    return adjectives

# Apply to male and female KWIC
female_adjs = extract_adjectives(female_kwic_all)
male_adjs = extract_adjectives(male_kwic_all)

# Count top adjectives
print("Top adjectives near 'woman/women':")
print(Counter(female_adjs).most_common(10))

print("\nTop adjectives near 'man/men':")
print(Counter(male_adjs).most_common(10))

target_nouns = {"man", "men", "woman", "women", "male", "female"}

def extract_modifying_adjectives(kwic_list, targets):
    modifiers = []
    for context in kwic_list:
        doc = nlp(context.replace("KWIC_", ""))
        for token in doc:
            # Look for target nouns
            if token.text.lower() in targets:
                # Check for adjective children (modifiers)
                for child in token.lefts:
                    if child.pos_ == "ADJ":
                        modifiers.append(child.text.lower())
    return modifiers

adjs_women = extract_modifying_adjectives(female_kwic_all, {"woman", "women", "female", "females"})
adjs_men = extract_modifying_adjectives(male_kwic_all, {"man", "men", "male", "males"})

# Count & display
from collections import Counter
print("Adjectives modifying 'woman/women':")
print(Counter(adjs_women).most_common(10))

print("\nAdjectives modifying 'man/men':")
print(Counter(adjs_men).most_common(10))

def plot_adj_dist(adj_counter, title):
    top = adj_counter.most_common(10)
    words, counts = zip(*top)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title(title)
    plt.xlabel("Adjective")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_adj_dist(Counter(adjs_women), "Adjectives Describing 'Woman/Women'")
plot_adj_dist(Counter(adjs_men), "Adjectives Describing 'Man/Men'")


