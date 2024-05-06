import requests
from bs4 import BeautifulSoup
import os
import csv
from datetime import datetime
import unicodedata

"""**Function to scrape articles links from a given URL**"""


def scrape_articles_links(url):
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'  # Set the encoding to UTF-8
        soup = BeautifulSoup(response.text, 'html.parser')
        posts_div = soup.find('div', class_='search-main')
        
        # Initialize a list to store links
        links = []

        # Find all the <div> tags with class "content-box"
        if posts_div:
            for div in posts_div.find_all('span', class_='read-more'):
                # Find all the <a> tags within the div
                for link in div.find_all('a'):
                    # Get the href attribute of each <a> tag
                    href = link.get('href')
                    # Append the link to the list
                    links.append(href)
        return links

    except Exception as e:
        print(f"Error scraping links from {url}: {e}")
        return None


"""**Function to process links**"""

def process_links(articles_links):
    unique_links = set()
    for link in articles_links:
        full_url = "https://ssec.si.edu/" + link
        unique_links.add(full_url)
    return list(unique_links)


"""**Function to scrape data from a given URL**"""


def scrape_data(url):
    article = {}
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = soup.find('div', class_='blog-content-wrapper')
        if article_content:
            article['link'] = url
            print(url)
            article_title_div  = article_content.find('div', class_='field-title-field')
            if article_title_div:
                h2_tag = article_title_div.find('h2')  # Find the <h2> tag within the div
                if h2_tag:
                    article_title = h2_tag.get_text(strip=True)  # Extract text from <h2> tag
                    article['title']=article_title
                else:
                    print("No <h2> tag found within the div")
                    article['title']=None

            body = article_content.find('div', class_='field-body')
            if body:
                text_elements = []

                # Iterate through the children of the article tag
                for child in body.children:
                    # Check if the child is a <p> tag
                    if child.name == 'p':
                        text_elements.append(child.get_text(strip=True))
                    # Check if the child is an <h2> tag
                    elif child.name == 'h2':
                        text_elements.append(child.get_text(strip=True))
                    # Check if the child is a <ul> tag
                    elif child.name == 'ul':
                        # Iterate through the <li> tags within the <ul> tag
                        for li in child.find_all('li'):
                            text_elements.append(li.get_text(strip=True))

                # Join the text elements into a single list
                text = ' '.join(text_elements)
            else:
                # Handle the case where the article body is not found
                text = None
            # Remove non-ASCII characters from the content
            article['content'] = ''.join(char for char in text if ord(char) < 128) if text else None
            if text:
                # Calculate the total number of words
                words_count = len(text.split())
            else:
                words_count = 0
            article['words_count'] = words_count
            # Get current datetime
            now = datetime.now()
            article['datetime'] = now.strftime("%Y-%m-%d %H:%M:%S")
        return article
    except Exception as e:
        print(f"Error scraping data from {url}: {e}")
        return None

"""**Function to save scraped data (dictionary) to a file as CSV format**"""

def save_to_csv(data, filename):
  try:
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
      fieldnames = ['datetime','label', 'title', 'link', 'content', 'words_count']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for article in data:
          writer.writerow({'datetime': article['datetime'],'label': article['label'], 'title': article['title'], 'link': article['link'], 'content': article['content'], 'words_count': article['words_count']})
    print(f"Data saved to {filename}")
  except Exception as e:
        print(f"Error saving data to {filename}: {e}")

"""**Function to scrape articles**"""

def scrape_articles(url_base, pages, min_articles):
    articles_data = []
    articles_count = 0
    for i in range(0, pages + 1):
        url = f"{url_base}?page/{i}"
        print("Page Link:" + url)
        articles_links = scrape_articles_links(url)
        links = process_links(articles_links)
        for link in links:
            data = scrape_data(link)
            if data and data.get('words_count', 0) > 500:
                articles_data.append({'index': articles_count + 1, 'label': 'science_education', **data})
                articles_count += 1
                if articles_count >= min_articles:
                    return articles_data
    return articles_data

"""Main function"""

# URL to scrape
edu_url = 'https://ssec.si.edu/stemvisions-blog'
pages = 4  # Number of pages to scrape
min_articles = 15  # Minimum number of articles to scrape

# Scrape articles
articles_data = scrape_articles(edu_url, pages, min_articles)
# Create a directory to save the articles
os.makedirs("articles", exist_ok=True)
csv_file = 'articles/science_education_articles.csv'


# Save to CSV file
save_to_csv(articles_data, csv_file)