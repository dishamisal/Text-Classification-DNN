'''
Beautiful Soup Parser to CSV

@author Disha Misal
'''

from bs4 import BeautifulSoup
import csv

# User Variables
html_file = "table.html"
output_csv = "soutput.csv"

html = open(html_file).read()
soup = BeautifulSoup(html)
table = soup.find("table")

output_rows = []
for table_row in table.findAll('tr'):
    columns = table_row.findAll('td')
    output_row = []
    for column in columns:
        output_row.append(column.text)
    output_rows.append(output_row)
print(output_rows)
    
with open(output_csv, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output_rows)
