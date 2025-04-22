#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:57:03 2025

@author: clarezureich
"""

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter 
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
#nltk.download('vader_lexicon')
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os


"""
Data Collection 
Part 1
Scraping Irish Times for articles with reference to Guinness
"""
#--------------Part 1 - Collecting all Articles' URLS --------------#


# --- CONFIG ---
EMAIL = "LOGIN EMAIL" #Replace with login email
PASSWORD = "LOGIN PASSWORD" #Replace with login password 
SEARCH_QUERY = "guinness" #Search query 
URL_OUTPUT = "/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/irish_times_guinness_urls.csv"
SCROLL_PAUSE = 1
MAX_SCROLLS = 1475 #10 news articles per scroll, 14,500 articles total 

# --- Setup Chrome ---
options = uc.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options)
driver.implicitly_wait(10)

# --- Login Flow ---
print("Logging in...")
driver.get("https://www.irishtimes.com")
try:
    WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
except: pass
try:
    WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[1]/button"))).click()
except: pass
WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    (By.XPATH, "//*[@id='main-nav']/div[1]/div[2]/div[2]/div[4]/button/span/div"))).click()
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "id_email_input"))).send_keys(EMAIL)
driver.find_element(By.ID, "id_password_input").send_keys(PASSWORD)
driver.find_element(By.XPATH, "//*[@id='fusion-app']/div[2]/header/div/div/div/div/div/form/fieldset/button").click()
WebDriverWait(driver, 10).until(EC.url_changes("https://www.irishtimes.com"))
print("Login successful!")

# --- Search for 'guinness' ---
print("Navigating to homepage and triggering search...")
driver.get("https://www.irishtimes.com")
search_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//*[@id='main-nav']/div[1]/div[1]/div[2]/button[1]"))
)
search_button.click()
time.sleep(2)

search_input = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//*[@id='queryly_query']"))
)
search_input.send_keys(SEARCH_QUERY)
time.sleep(5)

# --- Scroll Fully ---
print(f"Scrolling {MAX_SCROLLS} times to load all results...")
for i in range(MAX_SCROLLS):
    driver.execute_script("document.getElementById('queryly_resultscontainer').scrollIntoView(false);")
    driver.execute_script("window.scrollBy(0, window.innerHeight);")
    time.sleep(SCROLL_PAUSE)
    print(f"Scroll {i+1} completed")

# --- Wait for 1 minute to ensure all content is loaded ---
print("Waiting for 1 minute to allow all content to load...")
time.sleep(60)

# --- Collect All URLs Matching XPath Ending in '1' ---
print("Collecting all URLs matching XPath ending in '1'...")
urls = set()
containers = driver.find_elements(By.XPATH, "//*[@id='queryly_resultscontainer']/div")
for container in containers:
    a_tag = container.find_element(By.XPATH, ".//a[1]")  # Targeting the element with 'a[1]'
    href = a_tag.get_attribute("href")
    if href:
        urls.add(href.strip())  # Collect and add the matching URL

# --- Save to CSV ---
df = pd.DataFrame({"url": list(urls)})
df.to_csv(URL_OUTPUT, index=False)
print(f"Saved {len(urls)} unique URLs to {URL_OUTPUT}")

# --- Done ---
driver.quit()
print("URL collection complete")



#--------------Part 2 - Collect metadate from URL list --------------#

"""
Collects article metadata, including title, date, and body text, from the URLs.
Can run asynchronously in multiple terminals for speed 
"""

# --- CONFIG (redo, just in case) ---
EMAIL = "LOGIN EMAIL" #replace with login email
PASSWORD = "LOGIN PASSWORD" #replace with login email
INPUT_CSV =  "irish_times_guinness_urls.csv" #csv from Part 1 of script
OUTPUT_DIR = "scraped_chunks"
CHUNK_ID = 0        # Set to 0, 1, 2, 3, or 4 to run the different chunks of URLS (for processing speed)
TOTAL_CHUNKS = 3       # How many chunks to divide the dataset into
BATCH_SIZE = 200        # Save every x articles
TEST_COUNT = None      # set to an integer for partial run (to test script)
MIN_TEXT_LENGTH = 20   # minimum characters required to accept an article

# --- Prepare Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load URLs ---
df = pd.read_csv(INPUT_CSV)
urls = df['url'].dropna().unique().tolist()
chunked_urls = [urls[i::TOTAL_CHUNKS] for i in range(TOTAL_CHUNKS)]
urls = chunked_urls[CHUNK_ID]
if TEST_COUNT:
    urls = urls[:TEST_COUNT]

# --- Setup Chrome ---
options = uc.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--headless=new")
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options)
driver.implicitly_wait(5)

# --- Login Flow ---
print("\n Logging into Irish Times...")
driver.get("https://www.irishtimes.com")
time.sleep(2)

try:
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
    print("Accepted cookies.")
except:
    print("ℹNo cookie banner.")

try:
    WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[1]/button"))).click()
    print("Closed notification.")
except:
    print("No notification pop-up.")

WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='main-nav']/div[1]/div[2]/div[2]/div[4]/button/span/div"))).click()
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "id_email_input")))
driver.find_element(By.ID, "id_email_input").send_keys(EMAIL)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "id_password_input")))
driver.find_element(By.ID, "id_password_input").send_keys(PASSWORD)
driver.find_element(By.XPATH, "//*[@id='fusion-app']/div[2]/header/div/div/div/div/div/form/fieldset/button").click()
WebDriverWait(driver, 10).until(EC.url_changes("https://www.irishtimes.com"))
print("Logged in!")
time.sleep(2)

# --- Scrape Loop ---
batch_articles = []
skipped = []
batch_number = 0

for i, url in enumerate(urls):
    print(f"\n[{i+1}/{len(urls)}] Scraping: {url}")
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 3.5))
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "b-it-headline")))

        try:
            title = driver.find_element(By.CLASS_NAME, "b-it-headline").text.strip()
        except:
            title = "No title found"

        date = "No date found"
        try:
            date_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'date')] | //div[contains(@class, 'timestamp')] | //div[contains(@class, 'published')]")
            for d in date_elements:
                date_text = d.text.strip()
                if date_text:
                    date = date_text
                    break
        except:
            pass

        body_paras = driver.find_elements(By.CLASS_NAME, "c-paragraph")
        paywall_paras = driver.find_elements(By.CLASS_NAME, "paywall")
        body_text = " ".join(p.text.strip() for p in body_paras if p.text.strip())
        paywall_text = " ".join(p.text.strip() for p in paywall_paras if p.text.strip())
        full_text = f"{body_text} {paywall_text}".strip()

        if "Subscribe now" in full_text:
            full_text = full_text.split("Subscribe now")[0].strip()

        if not full_text or len(full_text) < MIN_TEXT_LENGTH:
            print(f"Too short ({len(full_text)} chars). Skipping.")
            skipped.append(url)
            continue

        batch_articles.append({
            "title": title,
            "date": date,
            "url": url,
            "full_text": full_text
        })

        print(f"Scraped: {title[:60]}...")

        if len(batch_articles) == BATCH_SIZE:
            output_path = os.path.join(OUTPUT_DIR, f"chunk_{CHUNK_ID}_batch_{batch_number}.csv")
            pd.DataFrame(batch_articles).to_csv(output_path, index=False)
            print(f"Saved batch {batch_number} with {len(batch_articles)} articles")
            batch_articles = []
            batch_number += 1

    except Exception as e:
        print(f"Skipping due to error: {e}")
        skipped.append(url)
        continue

# --- Final Save ---
if batch_articles:
    output_path = os.path.join(OUTPUT_DIR, f"chunk_{CHUNK_ID}_batch_{batch_number}.csv")
    pd.DataFrame(batch_articles).to_csv(output_path, index=False)
    print(f"Saved final batch {batch_number} with {len(batch_articles)} articles")

if skipped:
    pd.Series(skipped).to_csv(os.path.join(OUTPUT_DIR, f"chunk_{CHUNK_ID}_skipped.csv"), index=False)
    print(f"Skipped {len(skipped)} articles")

driver.quit()
print("Done scraping")



#--------------Part 3 - Combining, Cleaning, Rescraping previously misread articles --------------#
# --- Input from chunked CSVs directory ---
input_dir = '/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Code/scraped_chunks'

# Get all CSVs excluding ones with 'skipped'
files = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith('.csv') and 'skipped' not in f
]

# Load all CSVs into one DataFrame
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='ISO-8859-1')

df_list = [safe_read_csv(file) for file in sorted(files)]
combined_df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(combined_df)} rows from {len(files)} batch files.")

# --- Identify Paywalled / Misread Articles ---
paywalled_df = combined_df[combined_df['full_text'].str.strip() == "Want to keep reading?"]
misread_urls = paywalled_df['url'].dropna().unique().tolist()
#Filter out misread urls for now 
combined_df = combined_df[~combined_df['url'].isin(misread_urls)]

# --- Save to CSV for Rescraping ---
pd.Series(misread_urls).to_csv("misread_urls.csv", index=False)
print(f"Saved {len(misread_urls)} misread URLs to misread_urls.csv")




"""
Irish Times Misread URL Rerun Scraper
Applies Phase 2 scraping logic to misread_urls.csv after the initial round of scraping 
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import pandas as pd
import os

# --- CONFIG ---
EMAIL = "LOGIN EMAIL"  # replace with your login email
PASSWORD = "LOGIN PASSWORD"  # replace with your password
INPUT_CSV = "/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/misread_urls.csv"
OUTPUT_DIR = "rescued_chunks"
CHUNK_ID = 2          # Set to 0, 1, 2
TOTAL_CHUNKS = 3
BATCH_SIZE = 200
TEST_COUNT = None       # Set to integer to limit run
MIN_TEXT_LENGTH = 20

# --- Prepare Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load URLs ---
df = pd.read_csv(INPUT_CSV)
urls = df['urls'].dropna().unique().tolist()
chunked_urls = [urls[i::TOTAL_CHUNKS] for i in range(TOTAL_CHUNKS)]
urls = chunked_urls[CHUNK_ID]
if TEST_COUNT:
    urls = urls[:TEST_COUNT]

# --- Setup Chrome ---
options = uc.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--headless=new")
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options)
driver.implicitly_wait(5)

# --- Login Flow ---
print("\nLogging into Irish Times...")
driver.get("https://www.irishtimes.com")
time.sleep(2)

try:
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
    print("Accepted cookies.")
except:
    print("ℹ No cookie banner.")

try:
    WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[1]/button"))).click()
    print("Closed notification.")
except:
    print("No notification pop-up.")

WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='main-nav']/div[1]/div[2]/div[2]/div[4]/button/span/div"))).click()
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "id_email_input")))
driver.find_element(By.ID, "id_email_input").send_keys(EMAIL)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "id_password_input")))
driver.find_element(By.ID, "id_password_input").send_keys(PASSWORD)
driver.find_element(By.XPATH, "//*[@id='fusion-app']/div[2]/header/div/div/div/div/div/form/fieldset/button").click()
WebDriverWait(driver, 10).until(EC.url_changes("https://www.irishtimes.com"))
print("Logged in!")
time.sleep(2)

# --- Scrape Loop ---
batch_articles = []
skipped = []
batch_number = 0

for i, url in enumerate(urls):
    print(f"\n[{i+1}/{len(urls)}] Scraping: {url}")
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 3.5))
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "b-it-headline")))

        try:
            title = driver.find_element(By.CLASS_NAME, "b-it-headline").text.strip()
        except:
            title = "No title found"

        date = "No date found"
        try:
            date_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'date')] | //div[contains(@class, 'timestamp')] | //div[contains(@class, 'published')]")
            for d in date_elements:
                date_text = d.text.strip()
                if date_text:
                    date = date_text
                    break
        except:
            pass

        body_paras = driver.find_elements(By.CLASS_NAME, "c-paragraph")
        paywall_paras = driver.find_elements(By.CLASS_NAME, "paywall")
        body_text = " ".join(p.text.strip() for p in body_paras if p.text.strip())
        paywall_text = " ".join(p.text.strip() for p in paywall_paras if p.text.strip())
        full_text = f"{body_text} {paywall_text}".strip()

        if "Subscribe now" in full_text:
            full_text = full_text.split("Subscribe now")[0].strip()

        if not full_text or len(full_text) < MIN_TEXT_LENGTH:
            print(f"Too short ({len(full_text)} chars). Skipping.")
            skipped.append(url)
            continue

        batch_articles.append({
            "title": title,
            "date": date,
            "url": url,
            "full_text": full_text
        })

        print(f"Scraped: {title[:60]}...")

        if len(batch_articles) == BATCH_SIZE:
            output_path = os.path.join(OUTPUT_DIR, f"misread_chunk_{CHUNK_ID}_batch_{batch_number}.csv")
            pd.DataFrame(batch_articles).to_csv(output_path, index=False)
            print(f"Saved batch {batch_number} with {len(batch_articles)} articles")
            batch_articles = []
            batch_number += 1

    except Exception as e:
        print(f"Skipping due to error: {e}")
        skipped.append(url)
        continue

# --- Final Save ---
if batch_articles:
    output_path = os.path.join(OUTPUT_DIR, f"misread_chunk_{CHUNK_ID}_batch_{batch_number}.csv")
    pd.DataFrame(batch_articles).to_csv(output_path, index=False)
    print(f"Saved final batch {batch_number} with {len(batch_articles)} articles")

if skipped:
    pd.Series(skipped).to_csv(os.path.join(OUTPUT_DIR, f"misread_chunk_{CHUNK_ID}_skipped.csv"), index=False)
    print(f"Skipped {len(skipped)} articles")

driver.quit()
print("Done scraping misreads.")


##Recombine Rescraped URLS (that were properly scraped this time) with combined dataframe 
rescued_dir = '/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Code/rescued_chunks'

# Get all CSV files (excluding any 'skipped' ones)
rescued_files = [
    os.path.join(rescued_dir, f)
    for f in os.listdir(rescued_dir)
    if f.endswith('.csv') and 'skipped' not in f
]

# Load and concatenate all CSVs
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='ISO-8859-1')

rescued_list = [safe_read_csv(file) for file in sorted(rescued_files)]
rescued_df = pd.concat(rescued_list, ignore_index=True)

# Filter out rows with blank titles or invalid URLs
rescued_df = rescued_df[
    rescued_df['title'].notna() & 
    rescued_df['title'].str.strip().ne('') & 
    rescued_df['url'].str.startswith("https://www.irishtimes.com")
]

# Load the previously combined_df
scraped_dir = '/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Code/scraped_chunks'
scraped_files = [
    os.path.join(scraped_dir, f)
    for f in os.listdir(scraped_dir)
    if f.endswith('.csv') and 'skipped' not in f
]

scraped_list = [safe_read_csv(file) for file in sorted(scraped_files)]
combined_df = pd.concat(scraped_list, ignore_index=True)

# Drop any misreads from original combined_df
combined_df = combined_df[
    combined_df['full_text'].str.strip() != "Want to keep reading?"
]

#Drop any misreads from rescued_df
rescued_df['full_text'].str.strip() != "Want to keep reading?"

# Append the rescued data
final_df = pd.concat([combined_df, rescued_df], ignore_index=True)






"""
Data Analysis 
Part 2
Scraping Irish Times for articles with reference to Guinness
"""

###Preprocessing and Cleaning###

#df = pd.read_csv('/Users/clarezureich/Documents/Applied Social Data Science/QTA/final/Data Sets/irish_times_guinness_scraped.csv')

df = final_df 

### Drop non-articles (e.g. photography pages)###
df = df[~df['url'].str.contains('photography', case=False, na=False)]
df.head()

### Drop missing full_text ###
df = df[df['full_text'].str.strip() != '']
df = df.dropna(subset=['full_text'])

###Drop "Guinness World Record" Articles - not applicable to brand###
# Filter out articles that mention 'Guinness World Record' or 'Guinness World Records'
df = df[~df['title'].str.contains('Guinness World Record|Guinness World Records', case=False, na=False)]
df = df[~df['full_text'].str.contains('Guinness World Record|Guinness World Records', case=False, na=False)]


### Standardize date format - create a new table ###
df = df.copy()

# Build a new frame excluding “No date found”
df_dates = df[df['date'] != "No date found"].copy()

# Parse the dates on this new frame
df_dates.loc[:, 'date'] = pd.to_datetime(
    df_dates['date'],
    format='%a %b %d %Y - %H:%M',
    errors='coerce'
)

# Drop any rows that still failed to parse (should be 9488 remaining)
df_dates = df_dates.dropna(subset=['date']).copy()
print(len(df_dates), "rows after dropping unparseable dates")
df_dates['date'] = pd.to_datetime(
    df_dates['date'],
    format='%a %b %d %Y - %H:%M',
    errors='coerce'
)

#Extract year 
df_dates.loc[:, 'year'] = df_dates['date'].dt.year


### Clean and tokenize ###
def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

df_dates['tokens'] = df_dates['full_text'].apply(clean_tokenize)

# Lemmatize
def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_.isalpha()]

df_dates['lemmas'] = df_dates['tokens'].apply(lemmatize_tokens)
df_dates['cleaned_text'] = df_dates['lemmas'].apply(lambda x: ' '.join(x))


###Analysis###
#LDA for Topic Modeling
vectorizer = CountVectorizer(max_df=0.95, min_df=10, stop_words='english')
dtm = vectorizer.fit_transform(df_dates['cleaned_text'])

# Step 2: Fit LDA model
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(dtm)

# Step 3: Print topics
feature_names = vectorizer.get_feature_names_out()

def print_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"\nTopic #{topic_idx + 1}:")
        print(", ".join(top_words))

print_topics(lda_model, feature_names)



#Sentiment Analysis by VADER
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to the cleaned_text
df_dates['sentiment'] = df_dates['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize sentiment into positive, neutral, negative
def categorize_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df_dates['sentiment_label'] = df_dates['sentiment'].apply(categorize_sentiment)

# Preview sentiment results
print(df_dates[['title', 'date', 'sentiment', 'sentiment_label']].head())



#Visualizing sentiment over time 

# Group by year and calculate average sentiment
sentiment_by_year = df_dates.groupby('year')['sentiment'].mean()

# Plot sentiment over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=sentiment_by_year.index, y=sentiment_by_year.values, marker='o')
plt.title('Average Sentiment in Irish Times Articles Over Time')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.show()



# KWIC for gendered words
# Cleaned KWIC format to match old ad analysis
def kwic(text, keywords, window=5):
    tokens = text.lower().split()
    results = []
    for i, token in enumerate(tokens):
        if token in keywords:
            left = tokens[max(i - window, 0):i]
            right = tokens[i + 1:i + 1 + window]
            context = " ".join(left + [token] + right)  # no brackets
            results.append(context)
    return results

# Define keyword sets
female_terms = {"woman", "women", "lady", "ladies", "girl", "females", "female", "girl"}
male_terms = {"man", "men", "male", "barman", "gentleman", "lad", "lads", "boys", "boy"}

# Create a dictionary to store the KWIC results for both female and male terms
kwic_data = []
for idx, row in df_dates.iterrows():
    year = row['year']
    female_kwics = kwic(row['cleaned_text'], female_terms)
    male_kwics   = kwic(row['cleaned_text'], male_terms)

    for context in female_kwics:
        kwic_data.append({
            'gender': 'female',
            'year': year,
            'kwic_context': context,
            'article_idx': idx    # <-- original df index
        })
    for context in male_kwics:
        kwic_data.append({
            'gender': 'male',
            'year': year,
            'kwic_context': context,
            'article_idx': idx    # <-- original df index
        })

df_kwic = pd.DataFrame(kwic_data)

# Add sentiment using already-initialized VADER
df_kwic['sentiment'] = df_kwic['kwic_context'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_kwic['sentiment_label'] = df_kwic['sentiment'].apply(categorize_sentiment)

# Preview
df_kwic.head()




#Visualize Sentiment Analysis for Gendered KWIC 

#Sentiment distribution 
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_kwic, x='gender', y='sentiment')
plt.title("Sentiment Distribution of Gendered KWIC Contexts")
plt.grid(True)
plt.show()

#Postive/Neutral/Negative Counts 
plt.figure(figsize=(8, 5))
sns.countplot(data=df_kwic, x='gender', hue='sentiment_label')
plt.title("Sentiment Label Counts by Gendered KWIC Context")
plt.grid(True)
plt.show()



#Gendered Sentiment overtime 

# Filter for female and male KWICs
df_kwic_female = df_kwic[df_kwic['gender'] == 'female']
df_kwic_male = df_kwic[df_kwic['gender'] == 'male']


# Group by year and calculate average sentiment
sentiment_by_year_women = df_kwic_female.groupby('year')['sentiment'].mean()
sentiment_by_year_men = df_kwic_male.groupby('year')['sentiment'].mean()


# Plot sentiment over time for women
plt.figure(figsize=(10, 6))
sns.lineplot(x=sentiment_by_year_women.index, y=sentiment_by_year_women.values, marker='o')
plt.title('Sentiment of Women’s View of Guinness Over Time')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.show()

# Plot sentiment over time for men
plt.figure(figsize=(10, 6))
sns.lineplot(x=sentiment_by_year_men.index, y=sentiment_by_year_men.values, marker='o')
plt.title('Sentiment of Men’s View of Guinness Over Time')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.show()


#Top Verbs around Gendered Mentions 

#Parse KWIC Contexts 
df_kwic['doc'] = df_kwic['kwic_context'].apply(nlp)

# Helper: Find verbs in KWIC window
def extract_verbs(doc):
    return [token.lemma_ for token in doc if token.pos_ == 'VERB']

# Apply per gender
df_kwic['verbs'] = df_kwic['doc'].apply(extract_verbs)

# Separate by gender
verbs_female = [verb for sublist in df_kwic[df_kwic['gender'] == 'female']['verbs'] for verb in sublist]
verbs_male = [verb for sublist in df_kwic[df_kwic['gender'] == 'male']['verbs'] for verb in sublist]

#Filter out 'boring' verbs 
boring_verbs = {'say', 'make', 'get', 'take', 'have', 'be', 'go', 'come', 'do'}

# Then filter:
verbs_female = [v for v in verbs_female if v not in boring_verbs]
verbs_male = [v for v in verbs_male if v not in boring_verbs]

# Count top verbs
top_verbs_female = Counter(verbs_female).most_common(15)
top_verbs_male = Counter(verbs_male).most_common(15)

print("\nTop verbs near female mentions:")
print(top_verbs_female)

print("\nTop verbs near male mentions:")
print(top_verbs_male)

df_kwic.head()


# Focus Only on Verbs Attached to KWIC Token

#Helper that finds all verbs within ±window tokens of any gender term
def verbs_near_gender(doc, terms, window=5):
    verbs = []
    for i, tok in enumerate(doc):
        # look for the raw term
        if tok.text in terms:
            start = max(0, i - window)
            end   = min(len(doc), i + window + 1)
            for w in doc[start:end]:
                if w.pos_ == "VERB":
                    verbs.append(w.lemma_)
    return verbs

#Apply to df_kwic, picking the term set based on gender
df_kwic['verbs_window'] = df_kwic.apply(
    lambda row: verbs_near_gender(
        row['doc'],
        female_terms  if row['gender']=="female" 
        else male_terms,
        window=5
    ),
    axis=1
)

# Flatten and filter out high‑freq “boring” verbs
boring = {'say','make','get','take','have','do','go','come','be'}
verbs_female = [
    v for sub in df_kwic[df_kwic.gender=='female']['verbs_window'] 
        for v in sub 
    if v not in boring
]
verbs_male   = [
    v for sub in df_kwic[df_kwic.gender=='male']['verbs_window'] 
        for v in sub 
    if v not in boring
]

#See top 15
top_female = Counter(verbs_female).most_common(15)
top_male   = Counter(verbs_male).most_common(15)

print("Top verbs near female mentions:\n", top_female)
print("Top verbs near male mentions:\n",   top_male)




#Top Adjectives around Gendered Mentions 

# Helper: Find adjectives in KWIC
def extract_adjectives(doc):
    return [token.lemma_ for token in doc if token.pos_ == 'ADJ']

# Apply
df_kwic['adjectives'] = df_kwic['doc'].apply(extract_adjectives)

# Split by gender
adjs_female = [adj for sublist in df_kwic[df_kwic['gender'] == 'female']['adjectives'] for adj in sublist]
adjs_male = [adj for sublist in df_kwic[df_kwic['gender'] == 'male']['adjectives'] for adj in sublist]

# Count top adjectives
top_adjs_female = Counter(adjs_female).most_common(15)
top_adjs_male = Counter(adjs_male).most_common(15)

print("\nTop adjectives near female mentions:")
print(top_adjs_female)

print("\nTop adjectives near male mentions:")
print(top_adjs_male)





#Assign Dominant Topic to Each Article 

# Get topic probabilities for each article
topic_distributions = lda_model.transform(dtm)

# Assign the most likely topic (index of highest probability)
df_dates['dominant_topic'] = topic_distributions.argmax(axis=1)

# Map Topics from df_dates to df_kwic Add back index mapping (if lost)
df_dates = df_dates.reset_index(drop=True)

# Merge in dominant_topic by matching article_idx → df’s index
df_kwic = df_kwic.merge(
    df_dates[['dominant_topic']],
    left_on   = 'article_idx',
    right_index=True,
    how       ='left'
)

# Quick sanity check
print(df_kwic[['article_idx','dominant_topic']].head())


# Crosstab of topic × gender
topic_gender_pct = (
    pd.crosstab(df_kwic['dominant_topic'], df_kwic['gender'], normalize='index')
    * 100
).round(1)

# Plot it
topic_gender_pct.plot(
    kind='bar', stacked=True, figsize=(10,6), colormap='coolwarm'
)
plt.title("Gender Distribution of KWIC Mentions by LDA Topic")
plt.xlabel("Topic Number")
plt.ylabel("Percent of Mentions")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()





# Make sure year is numeric and sorted
df_kwic['year'] = df_kwic['year'].astype(int)
df_kwic = df_kwic.sort_values('year')

# --- Average sentiment over time by gender ---
sentiment_ts = (
    df_kwic
    .groupby(['year','gender'])['sentiment']
    .mean()
    .unstack()    # columns=['female','male']
    .sort_index()
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=sentiment_ts, dashes=False, markers=True)
plt.title("Average KWIC Sentiment Over Time by Gender")
plt.xlabel("Year")
plt.ylabel("Mean Compound Sentiment")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()





#--------------Comparison between News Article and Archived Advert Analysis--------------#

#Combine dataframes 

df_dates.columns
merged_metadata.columns
df_kwic.columns


#Group and average sentiment per year 

news_sentiment = df_dates.groupby("year")["sentiment"].mean() 
ads_sentiment = merged_metadata.groupby("year")["vader_score"].mean()

plt.figure(figsize=(10, 5))
ads_sentiment.plot(label="Adverts (VADER)", marker='o')
news_sentiment.plot(label="Newspaper (VADER)", marker='x')
plt.title("Average Sentiment Over Time")
plt.xlabel("Year")
plt.ylabel("Average VADER Sentiment Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Common Gendered Adjectives

def extract_adjectives(adjective_series):
    all_adj = []
    for row in adjective_series.dropna():
        try:
            tokens = ast.literal_eval(row) if isinstance(row, str) else row
            all_adj.extend(tokens)
        except:
            continue
    return Counter(all_adj).most_common(10)

female_news_adj = extract_adjectives(df_kwic[df_kwic["gender"] == "female"]["adjectives"])
male_news_adj = extract_adjectives(df_kwic[df_kwic["gender"] == "male"]["adjectives"])

female_ads_adj = extract_adjectives(merged_metadata["kwic_female"])
male_ads_adj = extract_adjectives(merged_metadata["kwic_male"])

# Dominant Topic Distributions

news_topic_counts = df_dates["dominant_topic"].value_counts().sort_index()
ads_topic_counts = merged_metadata["topic"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(news_topic_counts.index - 0.2, news_topic_counts.values, width=0.4, label="Newspaper")
plt.bar(ads_topic_counts.index + 0.2, ads_topic_counts.values, width=0.4, label="Adverts")
plt.xlabel("Topic Number")
plt.ylabel("Count")
plt.title("Dominant Topic Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Gender Representation

news_gender = df_kwic["gender"].value_counts()
ads_gender = merged_metadata["gender_group"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(news_gender.index, news_gender.values, alpha=0.6, label="Newspaper")
plt.bar(ads_gender.index, ads_gender.values, alpha=0.6, label="Adverts", bottom=news_gender.values)
plt.title("Gender Mentions in Texts")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

female_news_adj, male_news_adj, female_ads_adj, male_ads_adj


# Plot only the advert gender proportions
plt.figure(figsize=(8, 5))
plt.bar(ads_gender.index, ads_gender.values, alpha=0.7, color="darkslateblue")
plt.title("Gender Mentions in Advertisements")
plt.ylabel("Proportion of Mentions")
plt.xlabel("Gender Group")
plt.tight_layout()
plt.show()





