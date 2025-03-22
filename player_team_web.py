from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import urllib.parse
from selenium.webdriver.common.keys import Keys
import pandas as pd
import re

def get_all_team_links(url="https://gol.gg/teams/list/season-S6/split-ALL/tournament-ALL/"):
    """
    Opens the teams page, scrapes the teams table,
    and returns a dictionary {team_name: team_link, ...}.
    """
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(url)
    time.sleep(3)  # wait for page to load
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
    team_links = {}

    try:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # Adjust the selector to match the teams table on the page
        table = soup.select_one("table.table_list.playerslist.tablesaw.trhover.tablesaw-swipe.tablesaw-sortable")
        if not table:
            print("Teams table not found on the page.")
            return team_links

        rows = table.find_all("tr")
        if len(rows) < 2:
            print("No team rows found in the table.")
            return team_links

        # Loop over all rows except the header row
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # The first cell contains the team name and link
            team_name = cells[0].get_text(strip=True)
            link_tag = cells[0].find("a")
            if link_tag and "href" in link_tag.attrs:
                relative_link = link_tag["href"]
                # Remove the leading '.' if present
                if relative_link.startswith("."):
                    relative_link = relative_link[1:] #remove the . at the beginning
                
                # Build the full URL
                team_link = "https://gol.gg/teams" + relative_link
                team_links[team_name] = team_link

    finally:
        driver.quit()

    return team_links

def get_team_rosters(links_dict, year):
    """
    Given a dictionary of {team_name: team_url},
    navigates to each team's page, finds the roster table,
    and extracts a list of player names.
    
    Returns a dictionary: { team_name: [player1, player2, ...], ... }
    """
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    team_rosters = {}

    try:
        for team_name, team_url in links_dict.items():
            print(f"Scraping roster for team: {team_name} -> {team_url}")
            driver.get(team_url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            team_name = team_name + " " + year
            # Locate the roster table by its classes
            roster_table = soup.select_one("table.table_list.footable.toggle-square-filled.footable-loaded.default")
            if not roster_table:
                print(f"No roster table found for {team_name}")
                team_rosters[team_name] = []
                continue

            # Parse each row to get player names
            player_names = []
            rows = roster_table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                # Based on your screenshot, the second cell might contain the player's name.
                # If the table columns are [ROLE, PLAYER, KDA, ...], then index 1 = player name.
                if len(cells) >= 2:
                    player_name = cells[1].get_text(strip=True)
                    # Make sure it's not an empty row or header row
                    if player_name:
                        player_names.append(player_name)

            team_rosters[team_name] = player_names

    finally:
        driver.quit()

    return team_rosters

