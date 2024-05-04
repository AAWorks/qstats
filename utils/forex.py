import requests
from bs4 import BeautifulSoup

def fetch_etf_tickers(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    tickers = []

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print('Failed to retrieve data:', response.status_code)
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all ticker symbols in the page
    table = soup.find('table', attrs={'class': 'W(100%)'})
    if not table:
        print('No table found on the page')
        return
    
    for row in table.find_all('tr')[1:]:  # skip header row
        cols = row.find_all('td')
        if cols and len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers.append(ticker)
        
        # Update URL to the next page's URL

    return tickers

def main():
    url = 'https://uk.finance.yahoo.com/currencies/?guccounter=1'
    forex_tickers = fetch_etf_tickers(url)
    with open("data/forex_tickers.txt", "w") as f:
        f.write("\n".join(forex_tickers))

if __name__ == "__main__":
    main()