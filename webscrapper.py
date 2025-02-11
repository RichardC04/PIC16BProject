from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

# Set up Selenium (Chrome in this example)
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get("https://lol.fandom.com/wiki/LCK/2024_Season/Spring_Season/Player_Statistics")

# Wait for JavaScript to load the content (adjust the sleep time as needed)
time.sleep(5)

# Get page source and parse it with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Locate the table
table = soup.select_one("table.wikitable.sortable.spstats.plainlinks.hoverable-rows.jquery-tablesorter")

if table:
    print("Table found!")
    
    # --- Method 1: Extract with BeautifulSoup ---
    print("\nExtracting data using BeautifulSoup:")
    rows = table.find_all('tr')
    table_data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if cell_texts:
            table_data.append(cell_texts)
    
    # Print each row of the extracted data
    #for row in table_data:
        #print(row)

    result = {}
    for i in range(5, len(table_data)):
        temp = {}
        for j in range(2, 21):
            temp[f'{table_data[4][j]}'] = table_data[i][j]
        result[f'{table_data[i][1]}'] = temp

    # print(result)
        
else:
    print("Table not found.")

driver.quit()

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(result, orient='index')

# Save DataFrame to CSV
df.to_csv(r"C:\Users\29833\Documents\UCLA\PIC_16B\Project\player_statistics.csv", index_label="Player")

# Load the CSV back (optional)
df_loaded = pd.read_csv("player_statistics.csv")

# Display the first few rows
print(df_loaded.head())

# print(df.head())

