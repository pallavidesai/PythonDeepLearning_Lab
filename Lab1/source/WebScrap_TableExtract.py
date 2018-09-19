# Code to download a web page that contains a table using Request library,
# then parse the page using Beautifusoup library. Saved all the information of the table in a TableContent file.

import urllib.request
import requests
from bs4 import BeautifulSoup
import os

# Define a variable and put in link on that
html = requests.get("https://www.fantasypros.com/nfl/reports/leaders/qb.php?year=2015")
# Parse the source code using the Beautiful Soup library and save the parsed code in a variable
soup = BeautifulSoup(html.content, "html.parser")

# Print out the table content of the page
print(soup.table.text)
# opening File read write mode
TableContent = open("TableContent.txt", "r+")
# Putting table contents in file
TableContent.write(str(soup.table.text))
