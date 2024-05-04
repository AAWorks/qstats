import requests
from bs4 import BeautifulSoup

def fetch_etf_tickers(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    tickers = []

    for i in range(1, 6):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print('Failed to retrieve data:', response.status_code)
            break

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all ticker symbols in the page
        table = soup.find('table', attrs={'class': 'W(100%)'})
        if not table:
            print('No table found on the page')
            break
        
        for row in table.find_all('tr')[1:]:  # skip header row
            cols = row.find_all('td')
            if cols and len(cols) > 0:
                ticker = cols[0].text.strip()
                tickers.append(ticker)
        
        # Update URL to the next page's URL
        url = f"https://finance.yahoo.com/etfs/?count=100&offset={i * 100}"

    return tickers

def main():
    url = 'https://finance.yahoo.com/etfs/?count=100'
    etf_tickers = fetch_etf_tickers(url)
    with open("data/etf_tickers.txt", "w") as f:
        f.write("\n".join(etf_tickers))

if __name__ == "__main__":
    main()
