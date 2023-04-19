import pandas as pd
from datetime import datetime
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("tesla_kaggle.csv")

# Keep only the "Tweet Text" and "Date & Time" columns
df = df[["Date & Time", "Tweet Text"]]

# Define a function to filter out invalid dates
def filter_valid_dates(date_str):
    try:
        datetime.strptime(date_str, '%B %d, %Y at %I:%M%p')
        return True
    except ValueError:
        return False

# Apply the filter function to the 'date' column
valid_dates_mask = df['Date & Time'].apply(filter_valid_dates)
df = df[valid_dates_mask]


# Convert date string to datetime object
df["Date & Time"] = pd.to_datetime(df["Date & Time"], format="%B %d, %Y at %I:%M%p")


# Format Date & Time in "yyyy-mm-dd" format
df["Date & Time"] = df["Date & Time"].dt.strftime("%Y-%m-%d")


# Remove URLs, mentions, and hashtags from the Tweet Text text
df["Tweet Text"] = df["Tweet Text"].str.replace(r"http\S+", "")
df["Tweet Text"] = df["Tweet Text"].str.replace(r"@\S+", "")
df["Tweet Text"] = df["Tweet Text"].str.replace(r"#\S+", "")

# Remove leading and trailing whitespace from the Tweet Text text
df["Tweet Text"] = df["Tweet Text"].str.strip()
df.reset_index(inplace=False)
df = df.rename(columns={"Tweet Text": "Tweet", "Date & Time" : "Date"})

df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by='Date', ascending=False)

# Write the cleaned data to a new CSV file
df.to_csv("cleaned_hashtag_tesla_TweetTexts.csv", index=False)
df.to_gbq("is3107-project-383009.Dataset.tslaTweetsKaggle", project_id="is3107-project-383009", if_exists='replace')
