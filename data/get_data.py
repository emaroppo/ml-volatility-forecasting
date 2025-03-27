import yfinance as yf
import json
import time
from tqdm import tqdm

# update json file with top companies in each sector ({sector:[industries]}-> {sector:[{industry: [companies]}]})

with open("data/sectors_industries.json", "r") as f:
    sectors = json.load(f)
    for sector in sectors.keys():
        sectors[sector] = {industry: [] for industry in sectors[sector]}
    for sector in tqdm(sectors.keys()):

        for industry in sectors[sector]:
            top_companies = yf.Industry(industry).top_companies
            if top_companies is None:
                print(f"Error with {industry}")
                continue
            print(f"{industry} - {len(top_companies)}")
            sectors[sector][industry] = list(top_companies.index)

with open("data/sectors_industries.json", "w") as f:
    json.dump(sectors, f, indent=4)
