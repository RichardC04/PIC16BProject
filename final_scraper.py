from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd
import re

def get_league_data(player="player", season="ALL", year="S14", role="TOP"):
    '''
    This is a more comprehensive version of the webscraper from previous presentations.
    Takes in player/team, season(the whole year, spring, summer), year(season 6-14), and player role
    returns a nested dictionary composed of each team/player: {stat_name:stat, stat_name:stat ...}
    '''
    website = ""
    if player == "player":
        website = f"https://gol.gg/players/list/season-{year}/split-{season}/tournament-ALL/"
    else:
        website = f"https://gol.gg/teams/list/season-{year}/split-{season}/tournament-ALL/"
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(website)

    extracted_data = {}

    try:
        #Wait a bit for the page to load fully
        time.sleep(3)
        #Type "LCK" into the Leagues input (Selectize control)
        league_input = driver.find_element(By.CSS_SELECTOR, ".selectize-control.multi .selectize-input input")
        league_input.send_keys("LCK")
        time.sleep(1)
        league_input.send_keys(Keys.ENTER)  # finalize selection
        time.sleep(2)

        #Click the "Refresh" button to apply the LCK filter
        refresh_button = driver.find_element(By.ID, "btn_refresh")
        refresh_button.click()
        time.sleep(5)
        if player == "player": #if input is player
            #set the role filter with role input
            hidden_role = driver.find_element(By.ID, "hiddenfieldrole")
            driver.execute_script("arguments[0].value = arguments[1];", hidden_role, role)
    
            #Submit the form again to filter by role
            form = driver.find_element(By.ID, "FilterForm")
            form.submit()
            time.sleep(5)

        #Parse the updated page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        #Locate the table containing wanted data
        table = soup.select_one("table.table_list.playerslist.tablesaw.trhover.tablesaw-swipe.tablesaw-sortable")
        if table:
            #Extract rows
            rows = table.find_all('tr')
            table_data = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                if cell_texts:
                    table_data.append(cell_texts)

            #Build a 2d dict from the table
            if len(table_data) > 1:
                headers = table_data[0]
                for i in range(1, len(table_data)):
                    row_key = table_data[i][0] + " " + year
                    row_dict = {}
                    if player == "player":
                        row_dict["Season"] = year
                    for j in range(1, len(headers)):
                        if j < len(table_data[i]):
                            row_dict[headers[j]] = table_data[i][j]
                    extracted_data[row_key] = row_dict
            else:
                print("No valid data rows found.")
        else:
            print("Could not find the table.")

    finally:
        #Close the browser
        driver.quit()

    return extracted_data




